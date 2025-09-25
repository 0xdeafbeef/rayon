//! Rayon-core houses the core stable APIs of Rayon.
//!
//! These APIs have been mirrored in the Rayon crate and it is recommended to use these from there.
//!
//! [`join()`] is used to take two closures and potentially run them in parallel.
//!   - It will run in parallel if task B gets stolen before task A can finish.
//!   - It will run sequentially if task A finishes before task B is stolen and can continue on task B.
//!
//! [`scope()`] creates a scope in which you can run any number of parallel tasks.
//! These tasks can spawn nested tasks and scopes, but given the nature of work stealing, the order of execution can not be guaranteed.
//! The scope will exist until all tasks spawned within the scope have been completed.
//!
//! [`spawn()`] add a task into the 'static' or 'global' scope, or a local scope created by the [`scope()`] function.
//!
//! [`ThreadPool`] can be used to create your own thread pools (using [`ThreadPoolBuilder`]) or to customize the global one.
//! Tasks spawned within the pool (using [`install()`][tpinstall], [`join()`][tpjoin], etc.) will be added to a deque,
//! where it becomes available for work stealing from other threads in the local thread pool.
//!
//! [tpinstall]: ThreadPool::install()
//! [tpjoin]: ThreadPool::join()
//!
//! # Global fallback when threading is unsupported
//!
//! Rayon uses `std` APIs for threading, but some targets have incomplete implementations that
//! always return `Unsupported` errors. The WebAssembly `wasm32-unknown-unknown` and `wasm32-wasi`
//! targets are notable examples of this. Rather than panicking on the unsupported error when
//! creating the implicit global thread pool, Rayon configures a fallback mode instead.
//!
//! This fallback mode mostly functions as if it were using a single-threaded "pool", like setting
//! `RAYON_NUM_THREADS=1`. For example, `join` will execute its two closures sequentially, since
//! there is no other thread to share the work. However, since the pool is not running independent
//! of the main thread, non-blocking calls like `spawn` may not execute at all, unless a lower-
//! priority call like `broadcast` gives them an opening. The fallback mode does not try to emulate
//! anything like thread preemption or `async` task switching, but `yield_now` or `yield_local`
//! can also volunteer execution time.
//!
//! Explicit `ThreadPoolBuilder` methods always report their error without any fallback.
//!
//! # Restricting multiple versions
//!
//! In order to ensure proper coordination between thread pools, and especially
//! to make sure there's only one global thread pool, `rayon-core` is actively
//! restricted from building multiple versions of itself into a single target.
//! You may see a build error like this in violation:
//!
//! ```text
//! error: native library `rayon-core` is being linked to by more
//! than one package, and can only be linked to by one package
//! ```
//!
//! While we strive to keep `rayon-core` semver-compatible, it's still
//! possible to arrive at this situation if different crates have overly
//! restrictive tilde or inequality requirements for `rayon-core`.  The
//! conflicting requirements will need to be resolved before the build will
//! succeed.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![warn(rust_2018_idioms)]

use std::any::Any;
use std::env;
use std::error::Error;
use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;

#[macro_use]
mod private;

mod broadcast;
mod job;
mod join;
mod latch;
mod registry;
mod scope;
mod sleep;
mod spawn;
mod thread_pool;
mod unwind;

mod compile_fail;
mod test;

pub use self::broadcast::{broadcast, spawn_broadcast, BroadcastContext};
pub use self::join::{join, join_context};
pub use self::registry::ThreadBuilder;
pub use self::scope::{in_place_scope, scope, Scope};
pub use self::scope::{in_place_scope_fifo, scope_fifo, ScopeFifo};
pub use self::spawn::{spawn, spawn_fifo};
pub use self::thread_pool::current_thread_has_pending_tasks;
pub use self::thread_pool::current_thread_index;
pub use self::thread_pool::ThreadPool;
pub use self::thread_pool::{yield_local, yield_now, Yield};

#[cfg(not(feature = "web_spin_lock"))]
use std::sync;

#[cfg(feature = "web_spin_lock")]
use wasm_sync as sync;

use self::registry::{CustomSpawn, DefaultSpawn, ThreadSpawn};
use std::time::{Duration, Instant};

/// Receives notifications about scheduler state while a pool runs.
///
/// Recorders observe coarse worker lifecycle events. Implementations should be
/// lightweight, avoid blocking, and ideally remain panic-free. Panics are
/// caught so that Rayon stays operational, but doing so still disrupts pool
/// progress and should be treated as a last resort.
pub trait MetricsRecorder: Send + Sync + 'static {
    /// Called when a worker transitions between key states or when
    /// aggregate scheduler counters change.
    fn worker_event(&self, event: WorkerStateEvent);
}

impl<T> MetricsRecorder for T
where
    T: Fn(WorkerStateEvent) + Send + Sync + 'static,
{
    fn worker_event(&self, event: WorkerStateEvent) {
        (self)(event)
    }
}

/// Kind of worker transition that triggered an update.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerStateKind {
    /// Worker started searching for work.
    StartLooking,
    /// Worker located work and is transitioning to active.
    WorkFound,
    /// Worker announced it is sleepy (likely to sleep soon).
    Sleepy,
    /// Worker entered the blocked / sleeping state.
    Sleeping,
    /// Worker was explicitly woken.
    Woken,
    /// Worker resumed execution after completing the blocking wait.
    Resumed,
}

impl WorkerStateKind {
    /// Bit mask covering all defined kinds.
    pub const ALL_MASK: u64 = (1 << (Self::Resumed as u8 + 1)) - 1;

    /// Returns the bit mask representing this particular kind.
    pub const fn mask(self) -> u64 {
        1 << (self as u8)
    }
}

/// Describes the subset of [`WorkerStateKind`] transitions that should reach a recorder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct WorkerEventSet {
    mask: u64,
}

impl WorkerEventSet {
    /// Returns a set containing no events.
    pub const fn empty() -> Self {
        Self { mask: 0 }
    }

    /// Returns a set containing every currently-defined event.
    pub const fn all() -> Self {
        Self {
            mask: WorkerStateKind::ALL_MASK,
        }
    }

    /// Returns `true` if `kind` is enabled in this set.
    pub const fn contains(self, kind: WorkerStateKind) -> bool {
        self.mask & kind.mask() != 0
    }

    /// Returns the underlying bitmask.
    pub const fn mask(self) -> u64 {
        self.mask
    }

    fn from_iter_internal<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = WorkerStateKind>,
    {
        let mut mask = 0;
        for kind in iter {
            mask |= kind.mask();
        }
        Self {
            mask: mask & WorkerStateKind::ALL_MASK,
        }
    }
}

impl Default for WorkerEventSet {
    fn default() -> Self {
        Self::all()
    }
}

impl From<WorkerStateKind> for WorkerEventSet {
    fn from(kind: WorkerStateKind) -> Self {
        Self { mask: kind.mask() }
    }
}

impl From<&[WorkerStateKind]> for WorkerEventSet {
    fn from(kinds: &[WorkerStateKind]) -> Self {
        Self::from_iter_internal(kinds.iter().copied())
    }
}

impl<const N: usize> From<[WorkerStateKind; N]> for WorkerEventSet {
    fn from(kinds: [WorkerStateKind; N]) -> Self {
        Self::from_iter_internal(kinds)
    }
}

impl From<Vec<WorkerStateKind>> for WorkerEventSet {
    fn from(kinds: Vec<WorkerStateKind>) -> Self {
        Self::from_iter_internal(kinds)
    }
}

impl std::iter::FromIterator<WorkerStateKind> for WorkerEventSet {
    fn from_iter<I: IntoIterator<Item = WorkerStateKind>>(iter: I) -> Self {
        Self::from_iter_internal(iter)
    }
}

impl std::ops::BitOr for WorkerEventSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            mask: (self.mask | rhs.mask) & WorkerStateKind::ALL_MASK,
        }
    }
}

impl std::ops::BitOrAssign for WorkerEventSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.mask = (self.mask | rhs.mask) & WorkerStateKind::ALL_MASK;
    }
}

/// Snapshot of scheduler counters delivered to [`MetricsRecorder`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerStateEvent {
    /// Identifier of the pool that emitted the event.
    pub pool_id: u64,
    /// Index of the worker associated with this transition.
    pub worker_index: usize,
    /// Type of transition that occurred.
    pub kind: WorkerStateKind,
    /// Timestamp captured just before invoking the recorder.
    pub ts: Instant,
    /// Number of threads currently marked inactive (idle or sleeping).
    pub inactive_threads: usize,
    /// Number of inactive threads that are actively sleeping.
    pub sleeping_threads: usize,
    /// Number of threads that are awake but idle.
    pub awake_but_idle_threads: usize,
    /// Length of the emitting worker's local deque when the event was recorded.
    pub local_queue_depth: Option<usize>,
    /// Length of the global injector queue when the event was recorded.
    pub global_queue_depth: Option<usize>,
    /// Total number of LIFO spawns registered by this worker since pool start.
    pub lifo_spawn_count: usize,
    /// Total number of FIFO spawns registered by this worker since pool start.
    pub fifo_spawn_count: usize,
    /// Latency spent searching for work before a `WorkFound` event.
    pub search_latency: Option<Duration>,
    /// Time spent blocked while sleeping before a `Resumed` event.
    pub sleep_duration: Option<Duration>,
}

/// Returns the maximum number of threads that Rayon supports in a single thread pool.
///
/// If a higher thread count is requested by calling `ThreadPoolBuilder::num_threads` or by setting
/// the `RAYON_NUM_THREADS` environment variable, then it will be reduced to this maximum.
///
/// The value may vary between different targets, and is subject to change in new Rayon versions.
pub fn max_num_threads() -> usize {
    // We are limited by the bits available in the sleep counter's `AtomicUsize`.
    crate::sleep::THREADS_MAX
}

/// Returns the number of threads in the current registry. If this
/// code is executing within a Rayon thread pool, then this will be
/// the number of threads for the thread pool of the current
/// thread. Otherwise, it will be the number of threads for the global
/// thread pool.
///
/// This can be useful when trying to judge how many times to split
/// parallel work (the parallel iterator traits use this value
/// internally for this purpose).
///
/// # Future compatibility note
///
/// Note that unless this thread pool was created with a
/// builder that specifies the number of threads, then this
/// number may vary over time in future versions (see [the
/// `num_threads()` method for details][snt]).
///
/// [snt]: ThreadPoolBuilder::num_threads
pub fn current_num_threads() -> usize {
    crate::registry::Registry::current_num_threads()
}

/// Error when initializing a thread pool.
#[derive(Debug)]
pub struct ThreadPoolBuildError {
    kind: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    GlobalPoolAlreadyInitialized,
    CurrentThreadAlreadyInPool,
    IOError(io::Error),
}

/// Used to create a new [`ThreadPool`] or to configure the global rayon thread pool.
/// ## Creating a ThreadPool
/// The following creates a thread pool with 22 threads.
///
/// ```ignore-wasm
/// # use rayon_core as rayon;
/// let pool = rayon::ThreadPoolBuilder::new().num_threads(22).build().unwrap();
/// ```
///
/// To instead configure the global thread pool, use [`build_global()`]:
///
/// ```ignore-wasm
/// # use rayon_core as rayon;
/// rayon::ThreadPoolBuilder::new().num_threads(22).build_global().unwrap();
/// ```
///
/// [`build_global()`]: Self::build_global()
pub struct ThreadPoolBuilder<S = DefaultSpawn> {
    /// The number of threads in the rayon thread pool.
    /// If zero will use the RAYON_NUM_THREADS environment variable.
    /// If RAYON_NUM_THREADS is invalid or zero will use the default.
    num_threads: usize,

    /// The thread we're building *from* will also be part of the pool.
    use_current_thread: bool,

    /// Custom closure, if any, to handle a panic that we cannot propagate
    /// anywhere else.
    panic_handler: Option<Box<PanicHandler>>,

    /// Closure to compute the name of a thread.
    get_thread_name: Option<Box<dyn FnMut(usize) -> String>>,

    /// The stack size for the created worker threads
    stack_size: Option<usize>,

    /// Closure invoked on worker-thread start.
    start_handler: Option<Box<StartHandler>>,

    /// Closure invoked on worker-thread exit.
    exit_handler: Option<Box<ExitHandler>>,

    /// Closure invoked to spawn threads.
    spawn_handler: S,

    /// If false, worker threads will execute spawned jobs in a
    /// "depth-first" fashion. If true, they will do a "breadth-first"
    /// fashion. Depth-first is the default.
    breadth_first: bool,

    /// Optional metrics recorder that observes worker state transitions.
    metrics_recorder: Option<Arc<dyn MetricsRecorder>>,

    /// Filter describing which worker events reach the recorder.
    metrics_events: WorkerEventSet,
}

/// Contains the rayon thread pool configuration. Use [`ThreadPoolBuilder`] instead.
#[deprecated(note = "Use `ThreadPoolBuilder`")]
#[derive(Default)]
pub struct Configuration {
    builder: ThreadPoolBuilder,
}

/// The type for a panic-handling closure. Note that this same closure
/// may be invoked multiple times in parallel.
type PanicHandler = dyn Fn(Box<dyn Any + Send>) + Send + Sync;

/// The type for a closure that gets invoked when a thread starts. The
/// closure is passed the index of the thread on which it is invoked.
/// Note that this same closure may be invoked multiple times in parallel.
type StartHandler = dyn Fn(usize) + Send + Sync;

/// The type for a closure that gets invoked when a thread exits. The
/// closure is passed the index of the thread on which it is invoked.
/// Note that this same closure may be invoked multiple times in parallel.
type ExitHandler = dyn Fn(usize) + Send + Sync;

// NB: We can't `#[derive(Default)]` because `S` is left ambiguous.
impl Default for ThreadPoolBuilder {
    fn default() -> Self {
        ThreadPoolBuilder {
            num_threads: 0,
            use_current_thread: false,
            panic_handler: None,
            get_thread_name: None,
            stack_size: None,
            start_handler: None,
            exit_handler: None,
            spawn_handler: DefaultSpawn,
            breadth_first: false,
            metrics_recorder: None,
            metrics_events: WorkerEventSet::all(),
        }
    }
}

impl ThreadPoolBuilder {
    /// Creates and returns a valid rayon thread pool builder, but does not initialize it.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Note: the `S: ThreadSpawn` constraint is an internal implementation detail for the
/// default spawn and those set by [`spawn_handler`](#method.spawn_handler).
impl<S> ThreadPoolBuilder<S>
where
    S: ThreadSpawn,
{
    /// Creates a new `ThreadPool` initialized using this configuration.
    pub fn build(self) -> Result<ThreadPool, ThreadPoolBuildError> {
        ThreadPool::build(self)
    }

    /// Takes and returns the current metrics recorder, leaving `None`.
    fn take_metrics_recorder(&mut self) -> Option<Arc<dyn MetricsRecorder>> {
        self.metrics_recorder.take()
    }

    pub(crate) fn metrics_event_filter(&self) -> WorkerEventSet {
        self.metrics_events
    }

    /// Initializes the global thread pool. This initialization is
    /// **optional**.  If you do not call this function, the thread pool
    /// will be automatically initialized with the default
    /// configuration. Calling `build_global` is not recommended, except
    /// in two scenarios:
    ///
    /// - You wish to change the default configuration.
    /// - You are running a benchmark, in which case initializing may
    ///   yield slightly more consistent results, since the worker threads
    ///   will already be ready to go even in the first iteration.  But
    ///   this cost is minimal.
    ///
    /// Initialization of the global thread pool happens exactly
    /// once. Once started, the configuration cannot be
    /// changed. Therefore, if you call `build_global` a second time, it
    /// will return an error. An `Ok` result indicates that this
    /// is the first initialization of the thread pool.
    pub fn build_global(self) -> Result<(), ThreadPoolBuildError> {
        let registry = registry::init_global_registry(self)?;
        registry.wait_until_primed();
        Ok(())
    }
}

impl ThreadPoolBuilder {
    /// Creates a scoped `ThreadPool` initialized using this configuration.
    ///
    /// This is a convenience function for building a pool using [`std::thread::scope`]
    /// to spawn threads in a [`spawn_handler`].
    /// The threads in this pool will start by calling `wrapper`, which should
    /// do initialization and continue by calling `ThreadBuilder::run()`.
    ///
    /// [`spawn_handler`]: Self::spawn_handler()
    ///
    /// # Examples
    ///
    /// A scoped pool may be useful in combination with scoped thread-local variables.
    ///
    /// ```ignore-wasm
    /// # use rayon_core as rayon;
    ///
    /// scoped_tls::scoped_thread_local!(static POOL_DATA: Vec<i32>);
    ///
    /// fn main() -> Result<(), rayon::ThreadPoolBuildError> {
    ///     let pool_data = vec![1, 2, 3];
    ///
    ///     // We haven't assigned any TLS data yet.
    ///     assert!(!POOL_DATA.is_set());
    ///
    ///     rayon::ThreadPoolBuilder::new()
    ///         .build_scoped(
    ///             // Borrow `pool_data` in TLS for each thread.
    ///             |thread| POOL_DATA.set(&pool_data, || thread.run()),
    ///             // Do some work that needs the TLS data.
    ///             |pool| pool.install(|| assert!(POOL_DATA.is_set())),
    ///         )?;
    ///
    ///     // Once we've returned, `pool_data` is no longer borrowed.
    ///     drop(pool_data);
    ///     Ok(())
    /// }
    /// ```
    pub fn build_scoped<W, F, R>(self, wrapper: W, with_pool: F) -> Result<R, ThreadPoolBuildError>
    where
        W: Fn(ThreadBuilder) + Sync, // expected to call `run()`
        F: FnOnce(&ThreadPool) -> R,
    {
        std::thread::scope(|scope| {
            let pool = self
                .spawn_handler(|thread| {
                    let mut builder = std::thread::Builder::new();
                    if let Some(name) = thread.name() {
                        builder = builder.name(name.to_string());
                    }
                    if let Some(size) = thread.stack_size() {
                        builder = builder.stack_size(size);
                    }
                    builder.spawn_scoped(scope, || wrapper(thread))?;
                    Ok(())
                })
                .build()?;
            Ok(with_pool(&pool))
        })
    }
}

impl<S> ThreadPoolBuilder<S> {
    /// Sets a custom function for spawning threads.
    ///
    /// Note that the threads will not exit until after the pool is dropped. It
    /// is up to the caller to wait for thread termination if that is important
    /// for any invariants. For instance, threads created in [`std::thread::scope`]
    /// will be joined before that scope returns, and this will block indefinitely
    /// if the pool is leaked. Furthermore, the global thread pool doesn't terminate
    /// until the entire process exits!
    ///
    /// # Examples
    ///
    /// A minimal spawn handler just needs to call `run()` from an independent thread.
    ///
    /// ```ignore-wasm
    /// # use rayon_core as rayon;
    /// fn main() -> Result<(), rayon::ThreadPoolBuildError> {
    ///     let pool = rayon::ThreadPoolBuilder::new()
    ///         .spawn_handler(|thread| {
    ///             std::thread::spawn(|| thread.run());
    ///             Ok(())
    ///         })
    ///         .build()?;
    ///
    ///     pool.install(|| println!("Hello from my custom thread!"));
    ///     Ok(())
    /// }
    /// ```
    ///
    /// The default spawn handler sets the name and stack size if given, and propagates
    /// any errors from the thread builder.
    ///
    /// ```ignore-wasm
    /// # use rayon_core as rayon;
    /// fn main() -> Result<(), rayon::ThreadPoolBuildError> {
    ///     let pool = rayon::ThreadPoolBuilder::new()
    ///         .spawn_handler(|thread| {
    ///             let mut b = std::thread::Builder::new();
    ///             if let Some(name) = thread.name() {
    ///                 b = b.name(name.to_owned());
    ///             }
    ///             if let Some(stack_size) = thread.stack_size() {
    ///                 b = b.stack_size(stack_size);
    ///             }
    ///             b.spawn(|| thread.run())?;
    ///             Ok(())
    ///         })
    ///         .build()?;
    ///
    ///     pool.install(|| println!("Hello from my fully custom thread!"));
    ///     Ok(())
    /// }
    /// ```
    ///
    /// This can also be used for a pool of scoped threads like [`crossbeam::scope`],
    /// or [`std::thread::scope`] introduced in Rust 1.63, which is encapsulated in
    /// [`build_scoped`].
    ///
    /// [`crossbeam::scope`]: https://docs.rs/crossbeam/0.8/crossbeam/fn.scope.html
    /// [`build_scoped`]: Self::build_scoped()
    ///
    /// ```ignore-wasm
    /// # use rayon_core as rayon;
    /// fn main() -> Result<(), rayon::ThreadPoolBuildError> {
    ///     std::thread::scope(|scope| {
    ///         let pool = rayon::ThreadPoolBuilder::new()
    ///             .spawn_handler(|thread| {
    ///                 let mut builder = std::thread::Builder::new();
    ///                 if let Some(name) = thread.name() {
    ///                     builder = builder.name(name.to_string());
    ///                 }
    ///                 if let Some(size) = thread.stack_size() {
    ///                     builder = builder.stack_size(size);
    ///                 }
    ///                 builder.spawn_scoped(scope, || {
    ///                     // Add any scoped initialization here, then run!
    ///                     thread.run()
    ///                 })?;
    ///                 Ok(())
    ///             })
    ///             .build()?;
    ///
    ///         pool.install(|| println!("Hello from my custom scoped thread!"));
    ///         Ok(())
    ///     })
    /// }
    /// ```
    pub fn spawn_handler<F>(self, spawn: F) -> ThreadPoolBuilder<CustomSpawn<F>>
    where
        F: FnMut(ThreadBuilder) -> io::Result<()>,
    {
        ThreadPoolBuilder {
            spawn_handler: CustomSpawn::new(spawn),
            // ..self
            num_threads: self.num_threads,
            use_current_thread: self.use_current_thread,
            panic_handler: self.panic_handler,
            get_thread_name: self.get_thread_name,
            stack_size: self.stack_size,
            start_handler: self.start_handler,
            exit_handler: self.exit_handler,
            breadth_first: self.breadth_first,
            metrics_recorder: self.metrics_recorder,
            metrics_events: self.metrics_events,
        }
    }

    /// Returns a reference to the current spawn handler.
    fn get_spawn_handler(&mut self) -> &mut S {
        &mut self.spawn_handler
    }

    /// Get the number of threads that will be used for the thread
    /// pool. See `num_threads()` for more information.
    fn get_num_threads(&self) -> usize {
        if self.num_threads > 0 {
            self.num_threads
        } else {
            let default = || {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            };

            match env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|s| usize::from_str(&s).ok())
            {
                Some(x @ 1..) => return x,
                Some(0) => return default(),
                _ => {}
            }

            // Support for deprecated `RAYON_RS_NUM_CPUS`.
            match env::var("RAYON_RS_NUM_CPUS")
                .ok()
                .and_then(|s| usize::from_str(&s).ok())
            {
                Some(x @ 1..) => x,
                _ => default(),
            }
        }
    }

    /// Get the thread name for the thread with the given index.
    fn get_thread_name(&mut self, index: usize) -> Option<String> {
        let f = self.get_thread_name.as_mut()?;
        Some(f(index))
    }

    /// Sets a closure which takes a thread index and returns
    /// the thread's name.
    pub fn thread_name<F>(mut self, closure: F) -> Self
    where
        F: FnMut(usize) -> String + 'static,
    {
        self.get_thread_name = Some(Box::new(closure));
        self
    }

    /// Sets the number of threads to be used in the rayon thread pool.
    ///
    /// If you specify a non-zero number of threads using this
    /// function, then the resulting thread pools are guaranteed to
    /// start at most this number of threads.
    ///
    /// If `num_threads` is 0, or you do not call this function, then
    /// the Rayon runtime will select the number of threads
    /// automatically. At present, this is based on the
    /// `RAYON_NUM_THREADS` environment variable (if set),
    /// or the number of logical CPUs (otherwise).
    /// In the future, however, the default behavior may
    /// change to dynamically add or remove threads as needed.
    ///
    /// **Future compatibility warning:** Given the default behavior
    /// may change in the future, if you wish to rely on a fixed
    /// number of threads, you should use this function to specify
    /// that number. To reproduce the current default behavior, you
    /// may wish to use [`std::thread::available_parallelism`]
    /// to query the number of CPUs dynamically.
    ///
    /// **Old environment variable:** `RAYON_NUM_THREADS` is a one-to-one
    /// replacement of the now deprecated `RAYON_RS_NUM_CPUS` environment
    /// variable. If both variables are specified, `RAYON_NUM_THREADS` will
    /// be preferred.
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Use the current thread as one of the threads in the pool.
    ///
    /// The current thread is guaranteed to be at index 0, and since the thread is not managed by
    /// rayon, the spawn and exit handlers do not run for that thread.
    ///
    /// Note that the current thread won't run the main work-stealing loop, so jobs spawned into
    /// the thread pool will generally not be picked up automatically by this thread unless you
    /// yield to rayon in some way, like via [`yield_now()`], [`yield_local()`], or [`scope()`].
    ///
    /// # Local thread pools
    ///
    /// Using this in a local thread pool means the registry will be leaked. In future versions
    /// there might be a way of cleaning up the current-thread state.
    pub fn use_current_thread(mut self) -> Self {
        self.use_current_thread = true;
        self
    }

    /// Returns a copy of the current panic handler.
    fn take_panic_handler(&mut self) -> Option<Box<PanicHandler>> {
        self.panic_handler.take()
    }

    /// Normally, whenever Rayon catches a panic, it tries to
    /// propagate it to someplace sensible, to try and reflect the
    /// semantics of sequential execution. But in some cases,
    /// particularly with the `spawn()` APIs, there is no
    /// obvious place where we should propagate the panic to.
    /// In that case, this panic handler is invoked.
    ///
    /// If no panic handler is set, the default is to abort the
    /// process, under the principle that panics should not go
    /// unobserved.
    ///
    /// If the panic handler itself panics, this will abort the
    /// process. To prevent this, wrap the body of your panic handler
    /// in a call to `std::panic::catch_unwind()`.
    pub fn panic_handler<H>(mut self, panic_handler: H) -> Self
    where
        H: Fn(Box<dyn Any + Send>) + Send + Sync + 'static,
    {
        self.panic_handler = Some(Box::new(panic_handler));
        self
    }

    /// Get the stack size of the worker threads
    fn get_stack_size(&self) -> Option<usize> {
        self.stack_size
    }

    /// Sets the stack size of the worker threads
    pub fn stack_size(mut self, stack_size: usize) -> Self {
        self.stack_size = Some(stack_size);
        self
    }

    /// **(DEPRECATED)** Suggest to worker threads that they execute
    /// spawned jobs in a "breadth-first" fashion.
    ///
    /// Typically, when a worker thread is idle or blocked, it will
    /// attempt to execute the job from the *top* of its local deque of
    /// work (i.e., the job most recently spawned). If this flag is set
    /// to true, however, workers will prefer to execute in a
    /// *breadth-first* fashion -- that is, they will search for jobs at
    /// the *bottom* of their local deque. (At present, workers *always*
    /// steal from the bottom of other workers' deques, regardless of
    /// the setting of this flag.)
    ///
    /// If you think of the tasks as a tree, where a parent task
    /// spawns its children in the tree, then this flag loosely
    /// corresponds to doing a breadth-first traversal of the tree,
    /// whereas the default would be to do a depth-first traversal.
    ///
    /// **Note that this is an "execution hint".** Rayon's task
    /// execution is highly dynamic and the precise order in which
    /// independent tasks are executed is not intended to be
    /// guaranteed.
    ///
    /// This `breadth_first()` method is now deprecated per [RFC #1],
    /// and in the future its effect may be removed. Consider using
    /// [`scope_fifo()`] for a similar effect.
    ///
    /// [RFC #1]: https://github.com/rayon-rs/rfcs/blob/main/accepted/rfc0001-scope-scheduling.md
    #[deprecated(note = "use `scope_fifo` and `spawn_fifo` for similar effect")]
    pub fn breadth_first(mut self) -> Self {
        self.breadth_first = true;
        self
    }

    fn get_breadth_first(&self) -> bool {
        self.breadth_first
    }

    /// Takes the current thread start callback, leaving `None`.
    fn take_start_handler(&mut self) -> Option<Box<StartHandler>> {
        self.start_handler.take()
    }

    /// Sets a callback to be invoked on thread start.
    ///
    /// The closure is passed the index of the thread on which it is invoked.
    /// Note that this same closure may be invoked multiple times in parallel.
    /// If this closure panics, the panic will be passed to the panic handler.
    /// If that handler returns, then startup will continue normally.
    pub fn start_handler<H>(mut self, start_handler: H) -> Self
    where
        H: Fn(usize) + Send + Sync + 'static,
    {
        self.start_handler = Some(Box::new(start_handler));
        self
    }

    /// Returns a current thread exit callback, leaving `None`.
    fn take_exit_handler(&mut self) -> Option<Box<ExitHandler>> {
        self.exit_handler.take()
    }

    /// Sets a callback to be invoked on thread exit.
    ///
    /// The closure is passed the index of the thread on which it is invoked.
    /// Note that this same closure may be invoked multiple times in parallel.
    /// If this closure panics, the panic will be passed to the panic handler.
    /// If that handler returns, then the thread will exit normally.
    pub fn exit_handler<H>(mut self, exit_handler: H) -> Self
    where
        H: Fn(usize) + Send + Sync + 'static,
    {
        self.exit_handler = Some(Box::new(exit_handler));
        self
    }

    /// Filters which metrics events are forwarded to the recorder.
    ///
    /// Accepts any type convertible into [`WorkerEventSet`], enabling ergonomic
    /// filters like `WorkerEventSet::from([WorkerStateKind::Sleeping])` or slices
    /// of [`WorkerStateKind`]. The default forwards every currently-defined
    /// transition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rayon_core as rayon;
    /// use rayon::{ThreadPoolBuilder, WorkerStateKind};
    ///
    /// let _pool = ThreadPoolBuilder::new()
    ///     .metrics_events([WorkerStateKind::WorkFound, WorkerStateKind::Resumed])
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn metrics_events<E>(mut self, events: E) -> Self
    where
        E: Into<WorkerEventSet>,
    {
        self.metrics_events = events.into();
        self
    }

    /// Installs a metrics recorder that observes worker state transitions.
    ///
    /// The recorder is invoked with [`WorkerStateEvent`] snapshots that embed an
    /// `Instant` timestamp, the pool identifier, and optional
    /// `search_latency`/`sleep_duration` measurements when those values exist
    /// (`WorkFound` and `Resumed`, respectively). Events also expose the worker's
    /// current local/global queue depths alongside monotonic FIFO and LIFO spawn
    /// counters. Implementations should avoid blocking and remain panic-free.
    /// Rayon catches recorder panics so the pool can continue running, but doing
    /// so still disrupts progress.
    ///
    /// Closures of type `Fn(WorkerStateEvent) + Send + Sync + 'static` implement
    /// [`MetricsRecorder`], so simple inline loggers are ergonomic:
    ///
    /// ```
    /// # use rayon_core as rayon;
    /// use rayon::{ThreadPoolBuilder, WorkerStateKind};
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    ///
    /// let resumed = AtomicUsize::new(0);
    /// let _pool = ThreadPoolBuilder::new()
    ///     .metrics_recorder(|event| {
    ///         if event.kind == WorkerStateKind::Resumed {
    ///             resumed.fetch_add(1, Ordering::Relaxed);
    ///         }
    ///     })
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn metrics_recorder<R>(mut self, recorder: R) -> Self
    where
        R: MetricsRecorder,
    {
        self.metrics_recorder = Some(Arc::new(recorder));
        self
    }

    /// Sets the metrics recorder using an `Arc` (or any value convertible into one).
    pub fn metrics_recorder_arc<R>(mut self, recorder: R) -> Self
    where
        R: Into<Arc<dyn MetricsRecorder>>,
    {
        self.metrics_recorder = Some(recorder.into());
        self
    }
}

#[allow(deprecated)]
impl Configuration {
    /// Creates and return a valid rayon thread pool configuration, but does not initialize it.
    pub fn new() -> Configuration {
        Configuration {
            builder: ThreadPoolBuilder::new(),
        }
    }

    /// Deprecated in favor of `ThreadPoolBuilder::build`.
    pub fn build(self) -> Result<ThreadPool, Box<dyn Error + 'static>> {
        self.builder.build().map_err(Box::from)
    }

    /// Deprecated in favor of `ThreadPoolBuilder::thread_name`.
    pub fn thread_name<F>(mut self, closure: F) -> Self
    where
        F: FnMut(usize) -> String + 'static,
    {
        self.builder = self.builder.thread_name(closure);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::num_threads`.
    pub fn num_threads(mut self, num_threads: usize) -> Configuration {
        self.builder = self.builder.num_threads(num_threads);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::panic_handler`.
    pub fn panic_handler<H>(mut self, panic_handler: H) -> Configuration
    where
        H: Fn(Box<dyn Any + Send>) + Send + Sync + 'static,
    {
        self.builder = self.builder.panic_handler(panic_handler);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::stack_size`.
    pub fn stack_size(mut self, stack_size: usize) -> Self {
        self.builder = self.builder.stack_size(stack_size);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::breadth_first`.
    pub fn breadth_first(mut self) -> Self {
        self.builder = self.builder.breadth_first();
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::start_handler`.
    pub fn start_handler<H>(mut self, start_handler: H) -> Configuration
    where
        H: Fn(usize) + Send + Sync + 'static,
    {
        self.builder = self.builder.start_handler(start_handler);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::exit_handler`.
    pub fn exit_handler<H>(mut self, exit_handler: H) -> Configuration
    where
        H: Fn(usize) + Send + Sync + 'static,
    {
        self.builder = self.builder.exit_handler(exit_handler);
        self
    }

    /// Sets the event mask that filters which worker events reach the recorder.
    pub fn metrics_events<E>(mut self, events: E) -> Configuration
    where
        E: Into<WorkerEventSet>,
    {
        self.builder = self.builder.metrics_events(events);
        self
    }

    /// Deprecated in favor of `ThreadPoolBuilder::metrics_recorder`.
    pub fn metrics_recorder<R>(mut self, recorder: R) -> Configuration
    where
        R: MetricsRecorder,
    {
        self.builder = self.builder.metrics_recorder(recorder);
        self
    }

    /// Sets the metrics recorder on the deprecated configuration API using an `Arc`.
    pub fn metrics_recorder_arc<R>(mut self, recorder: R) -> Configuration
    where
        R: Into<Arc<dyn MetricsRecorder>>,
    {
        self.builder = self.builder.metrics_recorder_arc(recorder);
        self
    }

    /// Returns a ThreadPoolBuilder with identical parameters.
    fn into_builder(self) -> ThreadPoolBuilder {
        self.builder
    }
}

impl ThreadPoolBuildError {
    fn new(kind: ErrorKind) -> ThreadPoolBuildError {
        ThreadPoolBuildError { kind }
    }

    fn is_unsupported(&self) -> bool {
        matches!(&self.kind, ErrorKind::IOError(e) if e.kind() == io::ErrorKind::Unsupported)
    }
}

const GLOBAL_POOL_ALREADY_INITIALIZED: &str =
    "The global thread pool has already been initialized.";

const CURRENT_THREAD_ALREADY_IN_POOL: &str =
    "The current thread is already part of another thread pool.";

impl Error for ThreadPoolBuildError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match self.kind {
            ErrorKind::GlobalPoolAlreadyInitialized => GLOBAL_POOL_ALREADY_INITIALIZED,
            ErrorKind::CurrentThreadAlreadyInPool => CURRENT_THREAD_ALREADY_IN_POOL,
            ErrorKind::IOError(ref e) => e.description(),
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            ErrorKind::GlobalPoolAlreadyInitialized | ErrorKind::CurrentThreadAlreadyInPool => None,
            ErrorKind::IOError(e) => Some(e),
        }
    }
}

impl fmt::Display for ThreadPoolBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::CurrentThreadAlreadyInPool => CURRENT_THREAD_ALREADY_IN_POOL.fmt(f),
            ErrorKind::GlobalPoolAlreadyInitialized => GLOBAL_POOL_ALREADY_INITIALIZED.fmt(f),
            ErrorKind::IOError(e) => e.fmt(f),
        }
    }
}

/// Deprecated in favor of `ThreadPoolBuilder::build_global`.
#[deprecated(note = "use `ThreadPoolBuilder::build_global`")]
#[allow(deprecated)]
pub fn initialize(config: Configuration) -> Result<(), Box<dyn Error>> {
    config.into_builder().build_global().map_err(Box::from)
}

impl<S> fmt::Debug for ThreadPoolBuilder<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ThreadPoolBuilder {
            ref num_threads,
            ref use_current_thread,
            ref get_thread_name,
            ref panic_handler,
            ref stack_size,
            ref start_handler,
            ref exit_handler,
            spawn_handler: _,
            ref breadth_first,
            ref metrics_recorder,
            ..
        } = *self;

        // Just print `Some(<closure>)` or `None` to the debug
        // output.
        struct ClosurePlaceholder;
        impl fmt::Debug for ClosurePlaceholder {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("<closure>")
            }
        }
        let get_thread_name = get_thread_name.as_ref().map(|_| ClosurePlaceholder);
        let panic_handler = panic_handler.as_ref().map(|_| ClosurePlaceholder);
        let start_handler = start_handler.as_ref().map(|_| ClosurePlaceholder);
        let exit_handler = exit_handler.as_ref().map(|_| ClosurePlaceholder);
        let metrics_recorder = metrics_recorder.as_ref().map(|_| ClosurePlaceholder);

        f.debug_struct("ThreadPoolBuilder")
            .field("num_threads", num_threads)
            .field("use_current_thread", use_current_thread)
            .field("get_thread_name", &get_thread_name)
            .field("panic_handler", &panic_handler)
            .field("stack_size", &stack_size)
            .field("start_handler", &start_handler)
            .field("exit_handler", &exit_handler)
            .field("breadth_first", &breadth_first)
            .field("metrics_events", &self.metrics_events)
            .field("metrics_recorder", &metrics_recorder)
            .finish()
    }
}

#[allow(deprecated)]
impl fmt::Debug for Configuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.builder.fmt(f)
    }
}

/// Provides the calling context to a closure called by `join_context`.
#[derive(Debug)]
pub struct FnContext {
    migrated: bool,

    /// disable `Send` and `Sync`, just for a little future-proofing.
    _marker: PhantomData<*mut ()>,
}

impl FnContext {
    #[inline]
    fn new(migrated: bool) -> Self {
        FnContext {
            migrated,
            _marker: PhantomData,
        }
    }
}

impl FnContext {
    /// Returns `true` if the closure was called from a different thread
    /// than it was provided from.
    #[inline]
    pub fn migrated(&self) -> bool {
        self.migrated
    }
}

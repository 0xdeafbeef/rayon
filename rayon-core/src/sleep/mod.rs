//! Code that decides when workers should go to sleep. See README.md
//! for an overview.

use crate::latch::CoreLatch;
use crate::sync::{Condvar, Mutex};
use crate::{MetricsRecorder, WorkerEventSet, WorkerStateEvent, WorkerStateKind};
use crossbeam_utils::CachePadded;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

mod counters;
pub(crate) use self::counters::THREADS_MAX;
use self::counters::{AtomicCounters, JobsEventCounter};

/// The `Sleep` struct is embedded into each registry. It governs the waking and sleeping
/// of workers. It has callbacks that are invoked periodically at significant events,
/// such as when workers are looping and looking for work, when latches are set, or when
/// jobs are published, and it either blocks threads or wakes them in response to these
/// events. See the [`README.md`] in this module for more details.
///
/// [`README.md`] README.md
pub(super) struct Sleep {
    /// One "sleep state" per worker. Used to track if a worker is sleeping and to have
    /// them block.
    worker_sleep_states: Vec<CachePadded<WorkerSleepState>>,

    counters: AtomicCounters,

    pool_id: u64,

    events: WorkerEventSet,

    metrics: Option<Arc<dyn MetricsRecorder>>,

    spawn_counts: Vec<CachePadded<SpawnCounters>>,
}

/// An instance of this struct is created when a thread becomes idle.
/// It is consumed when the thread finds work, and passed by `&mut`
/// reference for operations that preserve the idle state. (In other
/// words, producing one of these structs is evidence the thread is
/// idle.) It tracks state such as how long the thread has been idle.
pub(super) struct IdleState {
    /// What is worker index of the idle thread?
    worker_index: usize,

    /// How many rounds have we been circling without sleeping?
    rounds: u32,

    /// Once we become sleepy, what was the sleepy counter value?
    /// Set to `INVALID_SLEEPY_COUNTER` otherwise.
    jobs_counter: JobsEventCounter,

    search_started_at: Instant,
}

/// The "sleep state" for an individual worker.
#[derive(Default)]
struct WorkerSleepState {
    /// Set to true when the worker goes to sleep; set to false when
    /// the worker is notified or when it wakes.
    is_blocked: Mutex<bool>,

    condvar: Condvar,
}

const ROUNDS_UNTIL_SLEEPY: u32 = 32;
const ROUNDS_UNTIL_SLEEPING: u32 = ROUNDS_UNTIL_SLEEPY + 1;

struct SpawnCounters {
    lifo: AtomicUsize,
    fifo: AtomicUsize,
}

impl SpawnCounters {
    fn new() -> Self {
        Self {
            lifo: AtomicUsize::new(0),
            fifo: AtomicUsize::new(0),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum SpawnKind {
    Lifo,
    Fifo,
}

impl Sleep {
    pub(super) fn new(
        n_threads: usize,
        pool_id: u64,
        events: WorkerEventSet,
        metrics: Option<Arc<dyn MetricsRecorder>>,
    ) -> Sleep {
        assert!(n_threads <= THREADS_MAX);
        Sleep {
            worker_sleep_states: (0..n_threads).map(|_| Default::default()).collect(),
            counters: AtomicCounters::new(),
            pool_id,
            events,
            metrics,
            spawn_counts: (0..n_threads)
                .map(|_| CachePadded::new(SpawnCounters::new()))
                .collect(),
        }
    }

    #[inline]
    fn should_emit(&self, kind: WorkerStateKind) -> bool {
        self.metrics.is_some() && self.events.contains(kind)
    }

    #[inline]
    pub(super) fn record_spawn(&self, worker_index: usize, kind: SpawnKind) {
        let counters = &self.spawn_counts[worker_index];
        match kind {
            SpawnKind::Lifo => {
                counters.lifo.fetch_add(1, Ordering::Relaxed);
            }
            SpawnKind::Fifo => {
                counters.fifo.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    fn report(
        &self,
        worker_index: usize,
        kind: WorkerStateKind,
        local_queue_depth: Option<usize>,
        global_queue_depth: Option<usize>,
        search_latency: Option<Duration>,
        sleep_duration: Option<Duration>,
    ) {
        let Some(recorder) = &self.metrics else {
            return;
        };

        if !self.events.contains(kind) {
            return;
        }

        let counters = self.counters.load(Ordering::SeqCst);
        let event = WorkerStateEvent {
            pool_id: self.pool_id,
            worker_index,
            kind,
            ts: Instant::now(),
            inactive_threads: counters.inactive_threads(),
            sleeping_threads: counters.sleeping_threads(),
            awake_but_idle_threads: counters.awake_but_idle_threads(),
            local_queue_depth,
            global_queue_depth,
            lifo_spawn_count: self.spawn_counts[worker_index].lifo.load(Ordering::Relaxed),
            fifo_spawn_count: self.spawn_counts[worker_index].fifo.load(Ordering::Relaxed),
            search_latency,
            sleep_duration,
        };

        let _ = panic::catch_unwind(AssertUnwindSafe(|| recorder.worker_event(event)));
    }

    #[inline]
    pub(super) fn start_looking(
        &self,
        worker_index: usize,
        local_queue_depth: Option<usize>,
        global_queue_depth: Option<usize>,
    ) -> IdleState {
        self.counters.add_inactive_thread();
        self.report(
            worker_index,
            WorkerStateKind::StartLooking,
            local_queue_depth,
            global_queue_depth,
            None,
            None,
        );

        IdleState {
            worker_index,
            rounds: 0,
            jobs_counter: JobsEventCounter::DUMMY,
            search_started_at: Instant::now(),
        }
    }

    #[inline]
    pub(super) fn work_found(
        &self,
        idle_state: &mut IdleState,
        local_queue_depth: Option<usize>,
        global_queue_depth: Option<usize>,
    ) {
        let worker_index = idle_state.worker_index;
        // If we were the last idle thread and other threads are still sleeping,
        // then we should wake up another thread. Do so before we clear our
        // inactive slot so observers never see `sleeping > inactive`.
        let sleepers = self.counters.load(Ordering::SeqCst).sleeping_threads();
        let to_wake = sleepers.min(2);
        if to_wake > 0 {
            self.wake_any_threads(to_wake as u32);
        }

        // Now mark this worker as active again.
        let _ = self.counters.sub_inactive_thread();
        let search_latency = if self.should_emit(WorkerStateKind::WorkFound) {
            Some(idle_state.search_started_at.elapsed())
        } else {
            None
        };
        self.report(
            worker_index,
            WorkerStateKind::WorkFound,
            local_queue_depth,
            global_queue_depth,
            search_latency,
            None,
        );
        idle_state.search_started_at = Instant::now();
    }

    #[inline]
    pub(super) fn no_work_found(
        &self,
        idle_state: &mut IdleState,
        latch: &CoreLatch,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        if idle_state.rounds < ROUNDS_UNTIL_SLEEPY {
            thread::yield_now();
            idle_state.rounds += 1;
        } else if idle_state.rounds == ROUNDS_UNTIL_SLEEPY {
            idle_state.jobs_counter = self.announce_sleepy();
            idle_state.rounds += 1;
            self.report(
                idle_state.worker_index,
                WorkerStateKind::Sleepy,
                None,
                None,
                None,
                None,
            );
            thread::yield_now();
        } else if idle_state.rounds < ROUNDS_UNTIL_SLEEPING {
            idle_state.rounds += 1;
            thread::yield_now();
        } else {
            debug_assert_eq!(idle_state.rounds, ROUNDS_UNTIL_SLEEPING);
            self.sleep(idle_state, latch, has_injected_jobs);
        }
    }

    #[cold]
    fn announce_sleepy(&self) -> JobsEventCounter {
        self.counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_active)
            .jobs_counter()
    }

    #[cold]
    fn sleep(
        &self,
        idle_state: &mut IdleState,
        latch: &CoreLatch,
        has_injected_jobs: impl FnOnce() -> bool,
    ) {
        let worker_index = idle_state.worker_index;

        if !latch.get_sleepy() {
            return;
        }

        let sleep_state = &self.worker_sleep_states[worker_index];
        let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
        debug_assert!(!*is_blocked);

        // Our latch was signalled. We should wake back up fully as we
        // will have some stuff to do.
        if !latch.fall_asleep() {
            idle_state.wake_fully();
            return;
        }

        loop {
            let counters = self.counters.load(Ordering::SeqCst);

            // Check if the JEC has changed since we got sleepy.
            debug_assert!(idle_state.jobs_counter.is_sleepy());
            if counters.jobs_counter() != idle_state.jobs_counter {
                // JEC has changed, so a new job was posted, but for some reason
                // we didn't see it. We should return to just before the SLEEPY
                // state so we can do another search and (if we fail to find
                // work) go back to sleep.
                idle_state.wake_partly();
                latch.wake_up();
                return;
            }

            // Otherwise, let's move from IDLE to SLEEPING.
            if self.counters.try_add_sleeping_thread(counters) {
                break;
            }
        }

        // Successfully registered as asleep.

        // We have one last check for injected jobs to do. This protects against
        // deadlock in the very unlikely event that
        //
        // - an external job is being injected while we are sleepy
        // - that job triggers the rollover over the JEC such that we don't see it
        // - we are the last active worker thread
        std::sync::atomic::fence(Ordering::SeqCst);
        if has_injected_jobs() {
            drop(is_blocked);
            // If we see an externally injected job, then we have to 'wake
            // ourselves up'. (Ordinarily, `sub_sleeping_thread` is invoked by
            // the one that wakes us.)
            self.counters.sub_sleeping_thread();
        } else {
            // If we don't see an injected job (the normal case), then flag
            // ourselves as asleep and wait till we are notified.
            //
            // (Note that `is_blocked` is held under a mutex and the mutex was
            // acquired *before* we incremented the "sleepy counter". This means
            // that whomever is coming to wake us will have to wait until we
            // release the mutex in the call to `wait`, so they will see this
            // boolean as true.)
            *is_blocked = true;
            drop(is_blocked);
            let resume_timer = if self.should_emit(WorkerStateKind::Resumed) {
                Some(Instant::now())
            } else {
                None
            };

            self.report(
                worker_index,
                WorkerStateKind::Sleeping,
                None,
                None,
                None,
                None,
            );

            let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
            while *is_blocked {
                is_blocked = sleep_state.condvar.wait(is_blocked).unwrap();
            }
            drop(is_blocked);

            let sleep_duration = resume_timer.map(|start| start.elapsed());
            self.report(
                worker_index,
                WorkerStateKind::Resumed,
                None,
                None,
                None,
                sleep_duration,
            );
        }

        // Update other state:
        idle_state.wake_fully();
        latch.wake_up();
    }

    /// Notify the given thread that it should wake up (if it is
    /// sleeping).  When this method is invoked, we typically know the
    /// thread is asleep, though in rare cases it could have been
    /// awoken by (e.g.) new work having been posted.
    pub(super) fn notify_worker_latch_is_set(&self, target_worker_index: usize) {
        self.wake_specific_thread(target_worker_index);
    }

    /// Signals that `num_jobs` new jobs were injected into the thread
    /// pool from outside. This function will ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_injected_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // This fence is needed to guarantee that threads
        // as they are about to fall asleep, observe any
        // new jobs that may have been injected.
        std::sync::atomic::fence(Ordering::SeqCst);

        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Signals that `num_jobs` new jobs were pushed onto a thread's
    /// local deque. This function will try to ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary. However, this is not guaranteed: under certain
    /// race conditions, the function may fail to wake any new
    /// threads; in that case the existing thread should eventually
    /// pop the job.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_internal_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Common helper for `new_injected_jobs` and `new_internal_jobs`.
    #[inline]
    fn new_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // Read the counters and -- if sleepy workers have announced themselves
        // -- announce that there is now work available. The final value of `counters`
        // with which we exit the loop thus corresponds to a state when
        let counters = self
            .counters
            .increment_jobs_event_counter_if(JobsEventCounter::is_sleepy);
        let num_awake_but_idle = counters.awake_but_idle_threads();
        let num_sleepers = counters.sleeping_threads();

        if num_sleepers == 0 {
            // nobody to wake
            return;
        }

        // Promote from u16 to u32 so we can interoperate with
        // num_jobs more easily.
        let num_awake_but_idle = num_awake_but_idle as u32;
        let num_sleepers = num_sleepers as u32;

        // If the queue is non-empty, then we always wake up a worker
        // -- clearly the existing idle jobs aren't enough. Otherwise,
        // check to see if we have enough idle workers.
        if !queue_was_empty {
            let num_to_wake = Ord::min(num_jobs, num_sleepers);
            self.wake_any_threads(num_to_wake);
        } else if num_awake_but_idle < num_jobs {
            let num_to_wake = Ord::min(num_jobs - num_awake_but_idle, num_sleepers);
            self.wake_any_threads(num_to_wake);
        }
    }

    #[cold]
    fn wake_any_threads(&self, mut num_to_wake: u32) {
        if num_to_wake > 0 {
            for i in 0..self.worker_sleep_states.len() {
                if self.wake_specific_thread(i) {
                    num_to_wake -= 1;
                    if num_to_wake == 0 {
                        return;
                    }
                }
            }
        }
    }

    fn wake_specific_thread(&self, index: usize) -> bool {
        let sleep_state = &self.worker_sleep_states[index];

        let should_wake = {
            let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
            if *is_blocked {
                *is_blocked = false;
                true
            } else {
                false
            }
        };

        if should_wake {
            // When the thread went to sleep, it will have incremented this
            // value. Adjust it before notifying so observers never see more
            // sleepers than inactive threads.
            self.counters.sub_sleeping_thread();

            sleep_state.condvar.notify_one();

            self.report(index, WorkerStateKind::Woken, None, None, None, None);

            true
        } else {
            false
        }
    }
}

impl IdleState {
    fn wake_fully(&mut self) {
        self.rounds = 0;
        self.jobs_counter = JobsEventCounter::DUMMY;
        self.search_started_at = Instant::now();
    }

    fn wake_partly(&mut self) {
        self.rounds = ROUNDS_UNTIL_SLEEPY;
        self.jobs_counter = JobsEventCounter::DUMMY;
        self.search_started_at = Instant::now();
    }
}

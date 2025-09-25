use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon_core::{
    join, ThreadPool, ThreadPoolBuilder, WorkerEventSet, WorkerStateEvent, WorkerStateKind,
};

fn run_basic_workload(pool: &ThreadPool) {
    pool.install(|| {
        join(|| {}, || {});
    });
}

#[test]
fn metrics_records_basic_invariants() {
    let events: Arc<Mutex<Vec<WorkerStateEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let collector = {
        let events = Arc::clone(&events);
        move |event: WorkerStateEvent| {
            events.lock().unwrap().push(event);
        }
    };

    let pool = ThreadPoolBuilder::new()
        .num_threads(2)
        .metrics_recorder(collector)
        .build()
        .unwrap();

    run_basic_workload(&pool);

    drop(pool);

    let events = events.lock().unwrap();
    assert!(!events.is_empty(), "expected some metrics events");

    let mut pool_ids = HashSet::new();
    let mut last_lifo = 0;
    let mut last_fifo = 0;
    for event in events.iter() {
        assert!(event.pool_id != 0);
        assert_eq!(
            event.search_latency.is_some(),
            event.kind == WorkerStateKind::WorkFound
        );
        assert_eq!(
            event.sleep_duration.is_some(),
            event.kind == WorkerStateKind::Resumed
        );
        assert!(event.lifo_spawn_count >= last_lifo);
        assert!(event.fifo_spawn_count >= last_fifo);
        last_lifo = event.lifo_spawn_count;
        last_fifo = event.fifo_spawn_count;
        if matches!(
            event.kind,
            WorkerStateKind::StartLooking | WorkerStateKind::WorkFound
        ) {
            assert!(
                event.local_queue_depth.is_some(),
                "queue depth should be captured for {:?}",
                event.kind
            );
            assert!(
                event.global_queue_depth.is_some(),
                "global depth should be captured for {:?}",
                event.kind
            );
        }
        pool_ids.insert(event.pool_id);
    }
    assert_eq!(pool_ids.len(), 1, "should only observe a single pool id");
}

#[test]
fn metrics_recorder_panic_is_caught() {
    let panic_count = Arc::new(AtomicUsize::new(0));
    let recorder = {
        let panic_count = Arc::clone(&panic_count);
        move |event: WorkerStateEvent| {
            panic_count.fetch_add(1, Ordering::SeqCst);
            if event.kind == WorkerStateKind::StartLooking {
                panic!("test recorder panic");
            }
        }
    };

    let pool = ThreadPoolBuilder::new()
        .num_threads(2)
        .metrics_recorder(recorder)
        .build()
        .unwrap();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| run_basic_workload(&pool)));
    assert!(result.is_ok(), "recorder panics should be contained");

    assert!(panic_count.load(Ordering::SeqCst) > 0);
}

#[test]
fn metrics_pool_ids_are_unique_per_pool() {
    let events_a: Arc<Mutex<Vec<WorkerStateEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let pool_a = ThreadPoolBuilder::new()
        .num_threads(2)
        .metrics_recorder({
            let events = Arc::clone(&events_a);
            move |event: WorkerStateEvent| {
                events.lock().unwrap().push(event);
            }
        })
        .build()
        .unwrap();

    run_basic_workload(&pool_a);
    drop(pool_a);

    let events_b: Arc<Mutex<Vec<WorkerStateEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let pool_b = ThreadPoolBuilder::new()
        .num_threads(2)
        .metrics_recorder({
            let events = Arc::clone(&events_b);
            move |event: WorkerStateEvent| {
                events.lock().unwrap().push(event);
            }
        })
        .build()
        .unwrap();

    run_basic_workload(&pool_b);
    drop(pool_b);

    let pool_id_a = events_a
        .lock()
        .unwrap()
        .first()
        .map(|event| event.pool_id)
        .expect("pool A should emit at least one event");
    let pool_id_b = events_b
        .lock()
        .unwrap()
        .first()
        .map(|event| event.pool_id)
        .expect("pool B should emit at least one event");

    assert_ne!(pool_id_a, pool_id_b, "each pool should carry a unique id");
}

#[test]
fn metrics_event_filters_accept_arrays() {
    let events: Arc<Mutex<Vec<WorkerStateEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let filter = WorkerEventSet::from([WorkerStateKind::WorkFound]);

    let pool = ThreadPoolBuilder::new()
        .num_threads(2)
        .metrics_events(filter)
        .metrics_recorder({
            let events = Arc::clone(&events);
            move |event: WorkerStateEvent| {
                events.lock().unwrap().push(event);
            }
        })
        .build()
        .unwrap();

    run_basic_workload(&pool);
    drop(pool);

    let events = events.lock().unwrap();
    assert!(
        !events.is_empty(),
        "expected masked events to still record work"
    );
    for event in events.iter() {
        assert_eq!(event.kind, WorkerStateKind::WorkFound);
    }
}

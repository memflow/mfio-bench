#[cfg(feature = "hash-check")]
use crc::{Algorithm, Crc, CRC_32_ISCSI};
use criterion::{measurement::Measurement, *};
use futures::stream::{FuturesUnordered, StreamExt};
use mfio::backend::IoBackend;
use mfio::io::{Packet, PacketIoExt, Write};
use mfio_netfs::*;
use mfio_rt::*;
use rand::prelude::*;
#[cfg(feature = "hash-check")]
use std::collections::BTreeMap;
use std::fs::File;
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const LATENCIES: &[usize] = &[1, 2, 5, 10, 20, 50, 100];
const READ_SIZES: &[usize] = &[0x1, 0x10, 0x100, 0x1000, 0x10000];
const MB: usize = 0x10000;

#[derive(Clone, Copy)]
enum AxisMode {
    /// X-Axis is represented by simulated latency
    Latency { read_size: usize },
    /// X-axis is represented by read size
    ReadSize { latency: usize },
}

impl AxisMode {
    fn iterable(self) -> &'static [usize] {
        match self {
            Self::Latency { .. } => &LATENCIES,
            Self::ReadSize { .. } => &READ_SIZES,
        }
    }

    fn read_size(self, iterable: usize) -> usize {
        match self {
            Self::Latency { read_size } => read_size,
            Self::ReadSize { .. } => iterable,
        }
    }

    fn latency(self, iterable: usize) -> usize {
        match self {
            Self::Latency { .. } => iterable,
            Self::ReadSize { latency } => latency,
        }
    }

    fn num_chunks(self, iterable: usize) -> usize {
        (64 * MB) / self.read_size(iterable)
    }
}

impl core::fmt::Display for AxisMode {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::Latency { read_size } => write!(f, "latency_mode 0x{read_size:x}"),
            Self::ReadSize { latency } => write!(f, "size_mode {latency}ms"),
        }
    }
}

trait ReadStrategy {
    fn name(&self) -> String;
    fn axis_mode(&self) -> AxisMode;
    fn pos(&self, total_iters: usize, file_len: u64, read_size: usize) -> u64;
}

struct SeqStrategy(AxisMode);

impl ReadStrategy for SeqStrategy {
    fn name(&self) -> String {
        format!("Sequential {}", self.0)
    }

    fn axis_mode(&self) -> AxisMode {
        self.0
    }

    fn pos(&self, total_iters: usize, file_len: u64, read_size: usize) -> u64 {
        (total_iters * read_size) as u64 % (file_len - (read_size as u64 - 1))
    }
}

#[derive(Clone, Copy)]
struct RevStrategy(AxisMode);

impl ReadStrategy for RevStrategy {
    fn name(&self) -> String {
        format!("Reverse {}", self.0)
    }

    fn axis_mode(&self) -> AxisMode {
        self.0
    }

    fn pos(&self, total_iters: usize, file_len: u64, read_size: usize) -> u64 {
        file_len
            - read_size as u64
            - ((total_iters * read_size) as u64 % (file_len - (read_size as u64 - 1)))
    }
}

#[derive(Clone, Copy)]
struct RandStrategy {
    axis_mode: AxisMode,
    seed: u64,
}

impl RandStrategy {
    pub fn new(axis_mode: AxisMode) -> Self {
        let mut rng = rand::thread_rng();
        let seed = rng.gen::<u64>();
        Self { axis_mode, seed }
    }
}

impl ReadStrategy for RandStrategy {
    fn name(&self) -> String {
        format!("Random {}", self.axis_mode)
    }

    fn axis_mode(&self) -> AxisMode {
        self.axis_mode
    }

    fn pos(&self, total_iters: usize, file_len: u64, read_size: usize) -> u64 {
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(self.seed + total_iters as u64);
        rng.gen::<u64>() % (file_len - (read_size - 1) as u64)
    }
}

fn file_read(c: &mut Criterion) {
    env_logger::init();

    let mut set_latency_state = None;

    let args = std::env::args().collect::<Vec<_>>();

    if !args.contains(&"latency_mode".into()) {
        // Reset the latency, just in case
        let _ = set_latency(&mut set_latency_state, 0);

        for latency in [1] {
            let axis_mode = AxisMode::ReadSize { latency };

            for strategy in &[
                Box::new(SeqStrategy(axis_mode)) as Box<dyn ReadStrategy>,
                Box::new(RevStrategy(axis_mode)),
                Box::new(RandStrategy::new(axis_mode)),
            ] {
                for (name, file, remote) in &[("local", "vol/sample.img", false)] {
                    file_with_strategy(
                        c,
                        &**strategy,
                        if *remote {
                            Some((
                                &mut set_latency_state,
                                crate::set_latency as for<'a> fn(&'a mut _, _) -> _,
                            ))
                        } else {
                            None
                        },
                        name,
                        file,
                        NativeRt::builder()
                            .enable_all()
                            .build_each()
                            .into_iter()
                            .filter_map(|(a, b)| Some(a).zip(b.ok())),
                    );
                }
            }
        }
    }

    if !args.contains(&"size_mode".into()) {
        for read_size in [0x1, 0x100, 0x10000] {
            let axis_mode = AxisMode::Latency { read_size };

            for strategy in &[
                Box::new(SeqStrategy(axis_mode)) as Box<dyn ReadStrategy>,
                Box::new(RevStrategy(axis_mode)),
                Box::new(RandStrategy::new(axis_mode)),
            ] {
                // On windows, we expect the smb share to be on Z: drive.
                #[cfg(windows)]
                std::env::set_current_dir("Z:\\").unwrap();
                for (name, file, remote) in &[("smb", "smb/sample.img", true)] {
                    file_with_strategy(
                        c,
                        &**strategy,
                        if *remote {
                            Some((
                                &mut set_latency_state,
                                crate::set_latency as for<'a> fn(&'a mut _, _) -> _,
                            ))
                        } else {
                            None
                        },
                        name,
                        file,
                        NativeRt::builder()
                            .enable_all()
                            .build_each()
                            .into_iter()
                            .filter_map(|(a, b)| Some(a).zip(b.ok())),
                    );
                }
            }
        }
    }
}

fn set_latency(stream: &mut Option<TcpStream>, latency: usize) -> std::io::Result<()> {
    use std::io::Write;

    if stream.is_none() {
        let addr = std::env::var("SET_LATENCY_ADDR");
        let addr = addr.as_deref().unwrap_or("127.0.0.1:12345");
        *stream = Some(TcpStream::connect(addr)?);
    }

    let stream = stream.as_mut().unwrap();

    if stream.write(format!("{latency}\n").as_bytes()).is_err() {
        let addr = std::env::var("SET_LATENCY_ADDR");
        let addr = addr.as_deref().unwrap_or("127.0.0.1:12345");
        *stream = TcpStream::connect(addr)?;
        stream.write(format!("{latency}\n").as_bytes())?;
    }
    stream.flush()?;

    Ok(())
}

#[cfg(not(unix))]
fn drop_cache(_: &Path) -> std::io::Result<()> {
    // it's nontrivial to drop file cache on windows
    Ok(())
}

#[cfg(unix)]
fn drop_cache(path: &Path) -> std::io::Result<()> {
    std::process::Command::new("/usr/bin/env")
        .args([
            "dd",
            &format!("if={}", path.to_str().unwrap()),
            "iflag=nocache",
            "count=0",
        ])
        .output()
        .map(|_| ())
}

fn file_with_strategy<S>(
    c: &mut Criterion,
    strategy: &(impl ReadStrategy + ?Sized),
    mut set_latency: Option<(&mut S, fn(&mut S, usize) -> std::io::Result<()>)>,
    name: &str,
    path: &str,
    mfio_runtimes: impl Iterator<Item = (&'static str, NativeRt)>,
) {
    let mut group = c.benchmark_group(strategy.name());

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    let temp_path = Path::new(path);

    let file_len = std::fs::metadata(temp_path).unwrap().len();

    let axis_mode = strategy.axis_mode();
    let total_iters = &Arc::new(Mutex::new(0));

    #[cfg(feature = "hash-check")]
    const CRC: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);
    #[cfg(feature = "hash-check")]
    let checksums: &Mutex<BTreeMap<(u64, usize), u32>> = &Default::default();

    for &iterable in axis_mode.iterable() {
        group.throughput(Throughput::Bytes(iterable as u64));
        let size = axis_mode.read_size(iterable);
        let num_chunks = axis_mode.num_chunks(iterable);
        let latency = axis_mode.latency(iterable);

        drop_cache(temp_path).unwrap();
        if let Some((ref mut state, func)) = set_latency {
            func(state, latency).unwrap();
        }

        let mut bufs = vec![vec![0u8; size]; num_chunks];

        group.bench_function(BenchmarkId::new(format!("std {name}"), iterable), |b| {
            b.iter_custom(|iters| {
                let total_iters = total_iters.clone();
                let mut total_iters = total_iters.lock().unwrap();

                // Translate iters represented in iterable bytes, to iters represented by
                // size bytes. This is so, since iterable does not always represent bytes.
                let mut iters = (iters as usize * iterable + (size - 1)) / size;

                use std::io::{Read, Seek, SeekFrom};

                let mut elapsed = Duration::default();

                let mut file = File::open(temp_path).unwrap();

                while iters > 0 {
                    file.rewind().unwrap();

                    let start = Instant::now();

                    for b in bufs.iter_mut().take(iters as _) {
                        let pos = strategy.pos(*total_iters, file_len, size);
                        file.seek(SeekFrom::Start(pos)).unwrap();
                        file.read_exact(&mut b[..]).unwrap();
                        #[cfg(feature = "hash-check")]
                        {
                            let mut digest = CRC.digest();
                            digest.update(&b[..]);
                            checksums
                                .lock()
                                .unwrap()
                                .insert((pos, size), digest.finalize());
                        }
                        *total_iters += 1;
                        iters -= 1;
                    }

                    elapsed += start.elapsed();
                }

                elapsed
            });
        });
        println!("TOTAL {}", total_iters.lock().unwrap());
    }

    for (rt_name, rt) in mfio_runtimes {
        #[cfg(feature = "hash-check")]
        {
            *total_iters.lock().unwrap() = 0;
        }

        fn mfio_bench<S, R: Fs + IoBackend, M: Measurement<Value = Duration>>(
            group: &mut BenchmarkGroup<M>,
            rt: &R,
            total_iters: &Mutex<usize>,
            file_len: u64,
            strategy: &(impl ReadStrategy + ?Sized),
            axis_mode: AxisMode,
            set_latency: &mut Option<(&mut S, fn(&mut S, usize) -> std::io::Result<()>)>,
            temp_path: &Path,
            rt_name: &str,
            name: &str,
        ) {
            for &iterable in axis_mode.iterable() {
                // We hack around the throughput plotting by making one loop iteration represent this many
                // bytes. We then internally scale the iterations to proper value.
                group.throughput(Throughput::Bytes(iterable as u64));
                let size = axis_mode.read_size(iterable);
                let num_chunks = axis_mode.num_chunks(iterable);
                let latency = axis_mode.latency(iterable);

                drop_cache(temp_path).unwrap();
                if let Some((ref mut state, func)) = set_latency {
                    func(state, latency).unwrap();
                }

                use criterion::async_executor::AsyncExecutor;

                struct MfioRt<'a, R: IoBackend>(&'a R);

                impl<'a, R: IoBackend> AsyncExecutor for MfioRt<'a, R> {
                    fn block_on<T>(&self, fut: impl core::future::Future<Output = T>) -> T {
                        self.0.block_on(fut)
                    }
                }

                let bufs = &(0..num_chunks)
                    .map(|_| Packet::<Write>::new_buf(size))
                    .collect::<Vec<_>>();

                group.bench_function(
                    BenchmarkId::new(format!("mfio-{rt_name} {name}"), iterable),
                    |b| {
                        b.to_async(MfioRt(rt)).iter_custom(|iters| {
                            // We are cloning arcs of bufs. This may be dangerous to do when `reset_err` is
                            // called, but we are sure right here that we indeed only use the packets once.
                            let mut bufs = bufs.clone();

                            async move {
                                let mut total_iters = total_iters.lock().unwrap();

                                // Translate iters represented in iterable bytes, to iters represented by
                                // size bytes. This is so, since iterable does not always represent bytes.
                                let mut iters = (iters as usize * iterable + (size - 1)) / size;

                                let file = &rt
                                    .open(temp_path, OpenOptions::new().read(true))
                                    .await
                                    .unwrap();

                                let mut futures = FuturesUnordered::new();

                                let start = Instant::now();

                                loop {
                                    while iters > 0 {
                                        if let Some(buf) = bufs.pop() {
                                            iters -= 1;
                                            futures.push({
                                                // SAFETY: we have exclusive access to the buffer at the moment
                                                unsafe { buf.reset_err() };
                                                // Issue a direct read @ here, because we want to queue up multiple
                                                // reads and have them all finish concurrently.
                                                let pos =
                                                    strategy.pos(*total_iters, file_len, size);
                                                *total_iters += 1;
                                                async move { (pos, file.io(pos, buf).await) }
                                            })
                                        } else {
                                            break;
                                        }
                                    }

                                    // TODO: grab all elems without blocking
                                    #[cfg_attr(not(feature = "hash-chack"), allow(unused))]
                                    if let Some((pos, b)) = futures.next().await {
                                        #[cfg(feature = "hash-check")]
                                        {
                                            if let Some(&orig) =
                                                checksums.lock().unwrap().get(&(pos, size))
                                            {
                                                let mut digest = CRC.digest();
                                                let slice = b.simple_contiguous_slice().unwrap();
                                                digest.update(slice);
                                                let digest = digest.finalize();
                                                assert_eq!(
                                                    digest,
                                                    orig,
                                                    "Checksums do not match (sz - {size} {})",
                                                    slice.len()
                                                );
                                            }
                                        }
                                        bufs.push(b);
                                    } else {
                                        break;
                                    }
                                }

                                start.elapsed()
                            }
                        });
                    },
                );

                println!("TOTAL {}", total_iters.lock().unwrap());
            }
        }

        mfio_bench(
            &mut group,
            &rt,
            &total_iters,
            file_len,
            strategy,
            axis_mode,
            &mut set_latency,
            temp_path,
            rt_name,
            name,
        );

        if let Ok(addr) = std::env::var("MFIO_REMOTE_ADDR") {
            let rt = NetworkFs::with_fs(addr.parse().unwrap(), rt.into(), true).unwrap();

            mfio_bench(
                &mut group,
                &rt,
                &total_iters,
                file_len,
                strategy,
                axis_mode,
                &mut set_latency,
                temp_path,
                &format!("netfs-{rt_name}"),
                name,
            );
        }
    }

    #[cfg(target_os = "linux")]
    for &iterable in axis_mode.iterable() {
        group.throughput(Throughput::Bytes(iterable as u64));
        let size = axis_mode.read_size(iterable);
        let num_chunks = axis_mode.num_chunks(iterable);
        let latency = axis_mode.latency(iterable);

        use criterion::async_executor::AsyncExecutor;
        use glommio::io::BufferedFile;
        use glommio::LocalExecutor;

        struct GlommioExecutor(LocalExecutor);

        impl<'a> AsyncExecutor for &'a GlommioExecutor {
            fn block_on<T>(&self, fut: impl core::future::Future<Output = T>) -> T {
                self.0.run(fut)
            }
        }

        let glommio = GlommioExecutor(LocalExecutor::default());

        drop_cache(temp_path).unwrap();
        if let Some((ref mut state, func)) = set_latency {
            func(state, latency).unwrap();
        }

        let bufs = &Mutex::new(vec![vec![0u8; size]; num_chunks]);

        group.bench_function(BenchmarkId::new(format!("glommio {name}"), iterable), |b| {
            b.to_async(&glommio).iter_custom(|iters| {
                let total_iters = total_iters.clone();
                async move {
                    let mut total_iters = total_iters.lock().unwrap();

                    let mut bufs = bufs.lock().unwrap();

                    // Translate iters represented in iterable bytes, to iters represented by
                    // size bytes. This is so, since iterable does not always represent bytes.
                    let mut iters = (iters as usize * iterable + (size - 1)) / size;

                    let file = &BufferedFile::open(temp_path).await.unwrap();

                    let mut futures = FuturesUnordered::new();

                    let start = Instant::now();

                    loop {
                        while iters > 0 {
                            if let Some(buf) = bufs.pop() {
                                iters -= 1;
                                futures.push({
                                    let pos = strategy.pos(*total_iters, file_len, size);
                                    *total_iters += 1;
                                    async move {
                                        let _res = file.read_at(pos, buf.len()).await.unwrap();
                                        // Let's not copy the file data to give glommio some
                                        // advantage.
                                        //buf.copy_from_slice(&res[..]);
                                        buf
                                    }
                                })
                            } else {
                                break;
                            }
                        }

                        // TODO: grab all elems without blocking
                        if let Some(b) = futures.next().await {
                            bufs.push(b);
                        } else {
                            break;
                        }
                    }

                    start.elapsed()
                }
            });
        });
    }

    for &iterable in axis_mode.iterable() {
        group.throughput(Throughput::Bytes(iterable as u64));
        let size = axis_mode.read_size(iterable);
        let num_chunks = axis_mode.num_chunks(iterable);
        let latency = axis_mode.latency(iterable);

        drop_cache(temp_path).unwrap();
        if let Some((ref mut state, func)) = set_latency {
            func(state, latency).unwrap();
        }

        let bufs = &Mutex::new(vec![vec![0u8; size]; num_chunks]);

        group.bench_function(BenchmarkId::new(format!("tokio {name}"), iterable), |b| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter_custom(|iters| {
                    let total_iters = total_iters.clone();
                    async move {
                        use std::io::SeekFrom;
                        use tokio::io::{AsyncReadExt, AsyncSeekExt};
                        let mut total_iters = total_iters.lock().unwrap();

                        let mut bufs = bufs.lock().unwrap();

                        // Translate iters represented in iterable bytes, to iters represented by
                        // size bytes. This is so, since iterable does not always represent bytes.
                        let mut iters = (iters as usize * iterable + (size - 1)) / size;

                        let file = &mut tokio::fs::File::open(temp_path).await.unwrap();

                        let mut elapsed = Duration::default();

                        while iters > 0 {
                            file.rewind().await.unwrap();

                            let start = Instant::now();

                            for b in bufs.iter_mut().take(iters as _) {
                                let titers = *total_iters;
                                file.seek(SeekFrom::Start(strategy.pos(titers, file_len, size)))
                                    .await
                                    .unwrap();
                                file.read(b).await.unwrap();
                                *total_iters += 1;
                                iters -= 1;
                            }

                            elapsed += start.elapsed();
                        }

                        elapsed
                    }
                });
        });
    }

    // compio happens to get stuck somewhere in the middle on linux
    #[cfg(not(target_os = "linux"))]
    for &iterable in axis_mode.iterable() {
        group.throughput(Throughput::Bytes(iterable as u64));
        let size = axis_mode.read_size(iterable);
        let num_chunks = axis_mode.num_chunks(iterable);
        let latency = axis_mode.latency(iterable);

        drop_cache(temp_path).unwrap();
        if let Some((ref mut state, func)) = set_latency {
            func(state, latency).unwrap();
        }

        use compio::runtime::Runtime;
        use criterion::async_executor::AsyncExecutor;

        struct CompioRuntime(Runtime);

        impl<'a> AsyncExecutor for CompioRuntime {
            fn block_on<T>(&self, fut: impl core::future::Future<Output = T>) -> T {
                self.0.block_on(fut)
            }
        }

        let bufs = &Mutex::new(vec![vec![0u8; size]; num_chunks]);

        group.bench_function(BenchmarkId::new(format!("compio {name}"), iterable), |b| {
            b.to_async(CompioRuntime(Runtime::new().unwrap()))
                .iter_custom(|iters| {
                    let total_iters = total_iters.clone();
                    async move {
                        use compio::{fs::File, io::AsyncReadAtExt};
                        let mut total_iters = total_iters.lock().unwrap();

                        let mut bufs = bufs.lock().unwrap();

                        // Translate iters represented in iterable bytes, to iters represented by
                        // size bytes. This is so, since iterable does not always represent bytes.
                        let mut iters = (iters as usize * iterable + (size - 1)) / size;

                        let file = &File::open(temp_path).await.unwrap();

                        let mut futures = FuturesUnordered::new();

                        let start = Instant::now();

                        loop {
                            while iters > 0 {
                                if let Some(mut buf) = bufs.pop() {
                                    iters -= 1;
                                    futures.push({
                                        buf.clear();
                                        buf.reserve_exact(size);
                                        let pos = strategy.pos(*total_iters, file_len, size);
                                        *total_iters += 1;
                                        async move {
                                            let (_, buf) =
                                                file.read_exact_at(buf, pos).await.unwrap();
                                            buf
                                        }
                                    })
                                } else {
                                    break;
                                }
                            }

                            // TODO: grab all elems without blocking
                            if let Some(b) = futures.next().await {
                                bufs.push(b);
                            } else {
                                break;
                            }
                        }

                        start.elapsed()
                    }
                });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        //.plotting_backend(PlottingBackend::Plotters)
        .with_plots()
        .warm_up_time(std::time::Duration::from_millis(1000))
        .measurement_time(std::time::Duration::from_millis(5000));
    targets =
        file_read,
}
criterion_main!(benches);

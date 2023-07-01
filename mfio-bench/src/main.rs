use core::mem::MaybeUninit;
use criterion::*;
use mfio::traits::*;
use mfio_fs::*;
use rand::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[no_mangle]
static mut FH: *const mfio::stdeq::Seekable<FileWrapper, u64> = core::ptr::null();

const LATENCIES: [usize; 5] = [/*0, */ 1, 2, 5, 10, 20 /*, 50, 100*/];

trait ReadStrategy {
    fn name(&self) -> String;
    fn read_size(&self) -> usize;
    fn pos(&self, total_iters: usize, file_len: u64) -> u64;
}

struct SeqStrategy(usize);

impl ReadStrategy for SeqStrategy {
    fn name(&self) -> String {
        format!("Sequential 0x{:x}", self.0)
    }

    fn read_size(&self) -> usize {
        self.0
    }

    fn pos(&self, total_iters: usize, file_len: u64) -> u64 {
        (total_iters * self.0) as u64 % (file_len - (self.0 as u64 - 1))
    }
}

struct RevStrategy(usize);

impl ReadStrategy for RevStrategy {
    fn name(&self) -> String {
        format!("Reverse 0x{:x}", self.0)
    }

    fn read_size(&self) -> usize {
        self.0
    }

    fn pos(&self, total_iters: usize, file_len: u64) -> u64 {
        file_len
            - self.0 as u64
            - ((total_iters * self.0) as u64 % (file_len - (self.0 as u64 - 1)))
    }
}

struct RandStrategy {
    read_size: usize,
    seed: u64,
}

impl RandStrategy {
    pub fn new(read_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let seed = rng.gen::<u64>();
        Self { read_size, seed }
    }
}

impl ReadStrategy for RandStrategy {
    fn name(&self) -> String {
        format!("Random 0x{:x}", self.read_size)
    }

    fn read_size(&self) -> usize {
        self.read_size
    }

    fn pos(&self, total_iters: usize, file_len: u64) -> u64 {
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(self.seed + total_iters as u64);
        rng.gen::<u64>() % (file_len - (self.read_size - 1) as u64)
    }
}

fn file_read(c: &mut Criterion) {
    env_logger::init();

    for size in [0x1000] {
        for strategy in &[
            Box::new(SeqStrategy(size)) as Box<dyn ReadStrategy>,
            Box::new(RevStrategy(size)),
            Box::new(RandStrategy::new(size)),
        ] {
            for (name, file) in &[
                //("local", "vol/sample.img"),
                //("nfs", "nfs/sample.img"),
                ("smb", "smb/sample.img"),
            ] {
                file_with_strategy(c, &**strategy, name, file);
            }
        }
    }
}
fn file_with_strategy(
    c: &mut Criterion,
    strategy: &(impl ReadStrategy + ?Sized),
    name: &str,
    path: &str,
) {
    let mut group = c.benchmark_group(strategy.name());

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    const MB: usize = 0x10000;

    let temp_path = Path::new(path);

    let drop_cache = |path: &Path| {
        std::process::Command::new("/usr/bin/env")
            .args([
                "dd",
                &format!("if={}", path.to_str().unwrap()),
                "iflag=nocache",
                "count=0",
            ])
            .output()
    };

    let mut stream = std::net::TcpStream::connect("127.0.0.1:12345").unwrap();

    let mut set_latency = |latency: usize| {
        use std::io::Write;
        stream.write(format!("{latency}\n").as_bytes())?;
        stream.flush()?;
        std::io::Result::Ok(())
    };

    let file_len = std::fs::metadata(temp_path).unwrap().len();

    let size = strategy.read_size();

    let num_chunks = (64 * MB) / size;

    let total_iters = &Arc::new(Mutex::new(0));

    for latency in LATENCIES {
        // We hack around the throughput plotting by making one loop iteration represent this many
        // bytes. We then internally scale the iterations to proper value.
        group.throughput(Throughput::Bytes(latency as u64));

        drop_cache(temp_path).unwrap();
        set_latency(latency).unwrap();

        group.bench_function(BenchmarkId::new(format!("mfio {name}"), latency), |b| {
            b.iter_custom(|iters| {
                let total_iters = total_iters.clone();
                let mut total_iters = total_iters.lock().unwrap();

                let mut bufs = vec![vec![MaybeUninit::uninit(); size]; num_chunks];

                // Translate iters represented in latency bytes, to iters represented by size bytes
                let mut iters = (iters as usize * latency + (size - 1)) / size;

                let mut elapsed = Duration::default();

                NativeFs::default().run(|fs| async move {
                    let file = fs.open(temp_path, OpenOptions::new().read(true));
                    unsafe { FH = &file as *const _ };

                    while iters > 0 {
                        let mut output = vec![];
                        output.reserve(num_chunks);

                        let start = Instant::now();

                        let file = &file;

                        for b in bufs.iter_mut().take(iters as _) {
                            // Issue a direct read @ here, because we want to queue up multiple
                            // reads and have them all finish concurrently.
                            let pos = strategy.pos(*total_iters, file_len);
                            let fut = async move {
                                let fut = file.read_all(pos, &mut b[..]);
                                fut.await.unwrap();
                            };
                            output.push(fut);
                            *total_iters += 1;
                            iters -= 1;
                        }

                        let _ = futures::future::join_all(output).await;

                        elapsed += start.elapsed();
                    }

                    elapsed
                })
            });
        });

        println!("TOTAL {}", total_iters.lock().unwrap());
    }

    #[cfg(target_os = "linux")]
    for latency in LATENCIES {
        use glommio::io::BufferedFile;
        use glommio::LocalExecutor;

        struct GlommioExecutor(LocalExecutor);

        impl<'a> AsyncExecutor for &'a GlommioExecutor {
            fn block_on<T>(&self, fut: impl core::future::Future<Output = T>) -> T {
                self.0.run(fut)
            }
        }

        let glommio = GlommioExecutor(LocalExecutor::default());

        group.throughput(Throughput::Bytes(latency as u64));

        drop_cache(temp_path).unwrap();
        set_latency(latency).unwrap();

        group.bench_function(BenchmarkId::new(format!("glommio {name}"), latency), |b| {
            b.to_async(&glommio).iter_custom(|iters| async move {
                let total_iters = total_iters.clone();
                let mut total_iters = total_iters.lock().unwrap();

                let mut bufs = vec![vec![0u8; size]; num_chunks];

                // Translate iters represented in latency bytes, to iters represented by size bytes
                let mut iters = (iters as usize * latency + (size - 1)) / size;

                let file = &BufferedFile::open(temp_path).await.unwrap();

                let mut elapsed = Duration::default();

                while iters > 0 {
                    let mut output = vec![];
                    output.reserve(num_chunks);

                    let start = Instant::now();

                    for b in bufs.iter_mut().take(iters as _) {
                        let comp = async move {
                            let res = file
                                .read_at(strategy.pos(*total_iters, file_len), b.len())
                                .await
                                .unwrap();
                            b.copy_from_slice(&res[..]);
                        };
                        output.push(comp);
                        *total_iters += 1;
                        iters -= 1;
                    }

                    let _ = futures::future::join_all(output).await;

                    elapsed += start.elapsed();
                }

                elapsed
            });
        });
    }

    for latency in LATENCIES {
        let mut bufs = vec![vec![0u8; size]; num_chunks];
        group.throughput(Throughput::Bytes(latency as u64));

        drop_cache(temp_path).unwrap();
        set_latency(latency).unwrap();

        group.bench_function(BenchmarkId::new(format!("std {name}"), latency), |b| {
            b.iter_custom(|iters| {
                let total_iters = total_iters.clone();
                let mut total_iters = total_iters.lock().unwrap();

                // Translate iters represented in latency bytes, to iters represented by size bytes
                let mut iters = (iters as usize * latency + (size - 1)) / size;

                use std::io::{Read, Seek, SeekFrom};

                let mut elapsed = Duration::default();

                let mut file = File::open(temp_path).unwrap();

                while iters > 0 {
                    file.rewind().unwrap();

                    let start = Instant::now();

                    for b in bufs.iter_mut().take(iters as _) {
                        let pos = strategy.pos(*total_iters, file_len);
                        file.seek(SeekFrom::Start(pos)).unwrap();
                        file.read_exact(&mut b[..]).unwrap();
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

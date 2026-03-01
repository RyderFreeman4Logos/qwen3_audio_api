fn main() {
    #[cfg(feature = "tch-backend")]
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "linux" {
            // Keep CUDA-linked libtorch deps (e.g. torch_cuda) from being
            // dropped by the linker's default as-needed behavior.
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        }

        if target_os == "linux" && std::env::var("LIBTORCH").is_err() {
            panic!(
                "\n\
                 =====================================================\n\
                 LIBTORCH environment variable is not set.\n\
                 \n\
                 Download libtorch (PyTorch C++ distribution) from:\n\
                   https://pytorch.org/get-started/locally/\n\
                 \n\
                 Then set:\n\
                   export LIBTORCH=/path/to/libtorch\n\
                   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH\n\
                 \n\
                 Example (CPU):\n\
                   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip\n\
                   unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cpu.zip\n\
                   export LIBTORCH=$(pwd)/libtorch\n\
                 \n\
                 Example (CUDA 12.8):\n\
                   wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip\n\
                   unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip\n\
                   export LIBTORCH=$(pwd)/libtorch\n\
                 ====================================================="
            );
        }
    }
}

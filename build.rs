// Only does anything with `--features opencv`: compiles the OpenCV backend and
// links the four OpenCV libraries it needs, found through pkg-config.
fn main() {
    #[cfg(feature = "opencv")]
    opencv::build();
}

#[cfg(feature = "opencv")]
mod opencv {
    use std::process::Command;

    pub fn build() {
        println!("cargo:rerun-if-changed=src/hint/cv_opencv.cpp");
        let package = ["opencv5", "opencv4"]
            .into_iter()
            .find(|p| {
                Command::new("pkg-config")
                    .args(["--exists", p])
                    .status()
                    .is_ok_and(|s| s.success())
            })
            .expect("--features opencv needs OpenCV 4 or 5 visible to pkg-config");
        let flags = Command::new("pkg-config")
            .args(["--cflags", package])
            .output()
            .expect("run pkg-config");
        let mut build = cc::Build::new();
        build.cpp(true).std("c++17").file("src/hint/cv_opencv.cpp");
        for flag in String::from_utf8_lossy(&flags.stdout).split_whitespace() {
            if let Some(dir) = flag.strip_prefix("-I") {
                build.include(dir);
            }
        }
        build.compile("stochos_cv_opencv");
        // Link only what the backend uses, not every module pkg-config lists
        // (the full list drags in optional modules such as viz and its VTK deps).
        for lib in ["opencv_imgproc", "opencv_core"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

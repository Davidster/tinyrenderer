Following [this tutorial](https://github.com/ssloy/tinyrenderer/wiki/Lesson-0:-getting-started) but using Rust instead of cpp.

To benchmark it:
```sh
# install deps
cargo install flamegraph

# linux
cargo flamegraph
# macos
cargo flamegraph --root

open flamegraph.svg
```
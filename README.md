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

random thingy
```
ffmpeg -i in.mp4 -filter:v "crop=in_w/2:in_h/2:in_w/2:in_h/2" -c:a copy out.mp4
ffmpeg -i 2022-01-29\ 01-08-07.mkv -filter:v "crop=in_w/2:in_h:in_w/4:in_h" -c:a copy cropped.mp4
```
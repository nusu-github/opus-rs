| Profile | Before Rust/C | After Rust/C | Relative cost reduction | Cases >=1.10 before / after |
| --- | ---: | ---: | ---: | ---: |
| float | 1.181 | 1.050 | 11.1% | 8 / 4 |
| fixed | 1.217 | 1.069 | 12.1% | 11 / 4 |
| fixed-res24 | 1.156 | 1.108 | 4.2% | 8 / 7 |

| Additional suite | Cases | Before Rust/C | After Rust/C | Relative cost reduction | Cases >=1.10 before / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| float-expanded | 14 | 1.288 | 1.134 | 12.0% | 12 / 8 |
| float-plc | 6 | 1.135 | 1.022 | 9.9% | 4 / 1 |
| pfa | 4 | 1.475 | 1.297 | 12.1% | 4 / 4 |
| qext | 8 | 1.364 | 1.263 | 7.4% | 8 / 8 |

| Profile / suite | Case | Operation | Rust/C | C microseconds/frame | Rust microseconds/frame |
| --- | --- | --- | ---: | ---: | ---: |
| float | hybrid-stereo | decode | 1.252 | 92.0 | 115.2 |
| float | celt-mono | decode | 1.254 | 29.5 | 37.0 |
| float | celt-stereo | encode | 1.121 | 198.1 | 222.1 |
| float | celt-stereo | decode | 1.330 | 54.8 | 72.8 |
| fixed | silk-mono | decode | 1.119 | 16.0 | 17.9 |
| fixed | silk-stereo | decode | 1.134 | 33.6 | 38.1 |
| fixed | celt-mono | decode | 1.229 | 35.0 | 43.0 |
| fixed | celt-stereo | decode | 1.293 | 67.5 | 87.3 |
| fixed-res24 | silk-mono | encode | 1.112 | 313.5 | 348.7 |
| fixed-res24 | silk-mono | decode | 1.185 | 16.3 | 19.3 |
| fixed-res24 | silk-stereo | decode | 1.141 | 37.5 | 42.8 |
| fixed-res24 | hybrid-mono | decode | 1.118 | 37.4 | 41.8 |
| fixed-res24 | hybrid-stereo | decode | 1.319 | 79.4 | 104.7 |
| fixed-res24 | celt-mono | decode | 1.161 | 37.5 | 43.5 |
| fixed-res24 | celt-stereo | decode | 1.255 | 68.1 | 85.5 |
| float-expanded | celt-2.5ms | encode | 1.298 | 46.3 | 60.1 |
| float-expanded | celt-2.5ms | decode | 1.424 | 9.4 | 13.3 |
| float-expanded | celt-10ms | encode | 1.120 | 101.5 | 113.8 |
| float-expanded | celt-10ms | decode | 1.356 | 25.8 | 35.0 |
| float-expanded | celt-60ms | encode | 1.114 | 595.4 | 663.4 |
| float-expanded | celt-60ms | decode | 1.305 | 162.3 | 211.9 |
| float-expanded | celt-low-complexity | encode | 1.409 | 94.9 | 133.7 |
| float-expanded | celt-low-complexity | decode | 1.250 | 46.8 | 58.4 |
| float-plc | celt-mono | decode | 1.129 | 34.8 | 39.3 |
| pfa | celt-mono | encode | 1.123 | 128.1 | 143.9 |
| pfa | celt-mono | decode | 1.418 | 29.3 | 41.5 |
| pfa | celt-stereo | encode | 1.183 | 194.9 | 230.7 |
| pfa | celt-stereo | decode | 1.503 | 52.8 | 79.3 |
| qext | qext-48000-1 | encode | 1.109 | 159.9 | 177.4 |
| qext | qext-48000-1 | decode | 1.310 | 61.0 | 80.0 |
| qext | qext-48000-2 | encode | 1.539 | 267.4 | 411.7 |
| qext | qext-48000-2 | decode | 1.371 | 122.8 | 168.4 |
| qext | qext-96000-1 | encode | 1.164 | 301.8 | 351.4 |
| qext | qext-96000-1 | decode | 1.274 | 107.8 | 137.3 |
| qext | qext-96000-2 | encode | 1.238 | 488.4 | 604.7 |
| qext | qext-96000-2 | decode | 1.150 | 231.9 | 266.7 |

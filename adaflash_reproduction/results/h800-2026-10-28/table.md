# H800 throughput

Output tokens/s, mean ± sample SD, three timed repetitions.

| Dataset | Requests | AR tok/s | Fixed SD tok/s | AdaFlash tok/s | Ada / AR | Ada / fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 256 | 5012.4 ± 197.1 | 8273.4 ± 17.6 | 8889.3 ± 5.5 | 1.77× | 1.07× |
| math500 | 256 | 4426.9 ± 75.7 | 9769.0 ± 23.7 | 9795.8 ± 17.6 | 2.21× | 1.00× |
| humaneval | 256 | 4768.1 ± 8.9 | 7401.5 ± 1.2 | 8465.9 ± 25.2 | 1.78× | 1.14× |
| mbpp | 256 | 3530.0 ± 3.0 | 6944.8 ± 4.2 | 8161.5 ± 6.6 | 2.31× | 1.18× |
| mt-bench | 256 | 3821.0 ± 7.3 | 3688.1 ± 8.6 | 4956.5 ± 13.1 | 1.30× | 1.34× |

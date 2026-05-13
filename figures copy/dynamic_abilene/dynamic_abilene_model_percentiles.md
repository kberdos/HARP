# Dynamic Abilene Model Comparison

All metrics are normalized. The current metric is taken from the
three-way resilience evaluator so resilient temporal, vanilla temporal,
and snapshot baseline are compared through the same code path.

| Metric | Model | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | Resilient temporal | 1.0399 | 1.0313 | 1.0923 | 1.1156 | 1.1423 | 1.2023 |
| Current | Vanilla temporal | 1.0287 | 1.0284 | 1.0454 | 1.0542 | 1.0944 | 1.1171 |
| Current | Snapshot baseline | 1.0837 | 1.0839 | 1.1157 | 1.1705 | 1.2611 | 1.2800 |
| Combined | Resilient temporal | 1.8480 | 1.8429 | 1.9366 | 1.9809 | 2.0359 | 2.0622 |
| Combined | Vanilla temporal | 1.8355 | 1.8421 | 1.8743 | 1.8874 | 1.9355 | 2.0112 |
| Combined | Snapshot baseline | 1.9375 | 1.9438 | 1.9997 | 2.0978 | 2.2233 | 2.2923 |
| Expected failure | Resilient temporal | 2.3058 | 2.3400 | 2.4540 | 2.5087 | 2.6020 | 2.7291 |
| Expected failure | Vanilla temporal | 2.3391 | 2.3758 | 2.4718 | 2.5190 | 2.6332 | 2.8011 |
| Expected failure | Snapshot baseline | 2.4964 | 2.5404 | 2.6438 | 2.7269 | 2.9794 | 3.0629 |
| Worst failure | Resilient temporal | 4.1595 | 4.1252 | 4.3692 | 4.4623 | 4.5694 | 4.8090 |
| Worst failure | Vanilla temporal | 4.1149 | 4.1137 | 4.1814 | 4.2169 | 4.3776 | 4.4683 |
| Worst failure | Snapshot baseline | 4.3346 | 4.3356 | 4.4627 | 4.6820 | 5.0443 | 5.1199 |

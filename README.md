# DLinear

Change `num_workers` from to `0` for the repo to work out of the box.
```python
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
```
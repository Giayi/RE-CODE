## End-to-End Case Study：MBPP/5 Direct LLM 与 Reflexion

本段只整理 Direct LLM 与 Reflexion 的实验部分；metagpt+egrcm+soft_prqe 的错误修复过程不在本段补写范围内。

实验数据已写入：

`Output/end to end case study/mbpp5_direct_reflexion_gpt-4o-mini_data.json`

### 任务设置

数据集：`MBPP public test`

指定任务：`MBPP/5`

实际对应记录：此案例在既有 MBPP public test 产出文件中对应到 `task_id = MBPP/734`，内容与指定原始需求一致，entry point 为 `sum_Of_Subarray_Prod`。

模型：`gpt-4o-mini`

原始需求：

```text
Write a python function to find sum of products of all possible subarrays.

def sum_Of_Subarray_Prod(arr,n)
```

public test：

```python
assert candidate([1,2,3]) == 20
```

参考正解逻辑：

```python
def sum_Of_Subarray_Prod(arr, n):
    ans = 0
    res = 0
    i = n - 1
    while i >= 0:
        incr = arr[i] * (1 + res)
        ans += incr
        res = incr
        i -= 1
    return ans
```

这里的关键冲突是：题目 prompt 给出的函数签名包含 `arr, n` 两个参数，但 public test 实际只调用 `candidate([1,2,3])`，也就是只传入 `arr`。

### Direct LLM 中间结果

执行设置：

```bash
python main.py \
  --run_name "mbpp_direct" \
  --root_dir "../Output/LLM" \
  --public-test "./benchmarks/mbpp/mbpp_public_test.jsonl" \
  --mbpp-test "./benchmarks/mbpp/mbpp_test.jsonl" \
  --strategy "simple" \
  --language "py" \
  --model "gpt-4o-mini" \
  --pass_at_k 1 \
  --max_iters 10 \
  --verbose
```

第 1 步，输入 prompt：

```python
def sum_Of_Subarray_Prod(arr,n):
    """Write a python function to find sum of products of all possible subarrays."""
```

第 2 步，Direct LLM 一次生成代码：

```python
def sum_Of_Subarray_Prod(arr, n):
    """Write a python function to find sum of products of all possible subarrays."""
    total_sum = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            total_sum += product
    return total_sum
```

第 3 步，public evaluator 组合测试：

```python
def check(candidate):
    assert candidate([1,2,3]) == 20

check(sum_Of_Subarray_Prod)
```

第 4 步，评测错误：

```text
TypeError: sum_Of_Subarray_Prod() missing 1 required positional argument: 'n'
```

Direct LLM 最终结果：`Fail`

Direct LLM 最终代码：

```python
def sum_Of_Subarray_Prod(arr, n):
    """Write a python function to find sum of products of all possible subarrays."""
    total_sum = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            total_sum += product
    return total_sum
```

Direct LLM 失败原因：生成的算法本身可以计算 `[1, 2, 3]` 的子数组乘积总和，但函数接口不符合 public test。public test 只传入一个参数，程序要求两个参数，因此在执行到算法前就失败。

### Reflexion 中间结果

执行设置：

```bash
python main.py \
  --run_name "mbpp_reflexion" \
  --root_dir "../Output/Reflexion" \
  --public-test "./benchmarks/mbpp/mbpp_public_test.jsonl" \
  --mbpp-test "./benchmarks/mbpp/mbpp_test.jsonl" \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-4o-mini" \
  --pass_at_k 1 \
  --max_iters 10 \
  --verbose
```

第 1 步，Reflexion 初始生成代码：

```python
def sum_Of_Subarray_Prod(arr, n):
    """Write a python function to find sum of products of all possible subarrays."""
    total_sum = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            total_sum += product
    return total_sum
```

第 2 步，内部测试反馈：

```text
Tested passed:
assert sum_Of_Subarray_Prod([1, 2], 2) == 5      # Subarrays: [1], [2], [1,2]

Tests failed:
```

第 3 步，Reflection 记录：

```text
[]
```

这代表此案例没有留下任何反思内容。原因是内部测试使用了两个参数 `[1, 2], 2`，与模型生成的函数签名相容，因此没有暴露 public test 的调用方式错误。

第 4 步，Reflexion 最终代码：

```python
def sum_Of_Subarray_Prod(arr, n):
    """Write a python function to find sum of products of all possible subarrays."""
    total_sum = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            total_sum += product
    return total_sum
```

第 5 步，public evaluator 组合测试：

```python
def check(candidate):
    assert candidate([1,2,3]) == 20

check(sum_Of_Subarray_Prod)
```

第 6 步，评测错误：

```text
TypeError: sum_Of_Subarray_Prod() missing 1 required positional argument: 'n'
```

Reflexion 最终结果：`Fail`

Reflexion 失败原因：Reflexion 没有收到能揭露 public test 接口不一致的失败信号，因此没有产生修正方向。它保留了初始两参数版本，最后在 public test 中和 Direct LLM 一样因缺少 `n` 参数而失败。

### 对比结论

Direct LLM：`Fail`

Reflexion：`Fail`

两者的最终代码相同，失败点也相同：模型都遵循 prompt 中的 `def sum_Of_Subarray_Prod(arr,n)`，但 public test 实际要求可用 `candidate(arr)` 调用。这个案例适合用来展示单纯生成与 Reflexion 的限制：若测试反馈没有暴露真实评测的失败模式，Reflexion 不一定能启动有效修复。

对后续 metagpt+egrcm+soft_prqe 对比而言，这个失败案例的核心修复目标应该是把函数接口改成能符合 public test，例如让 `n` 可由 `len(arr)` 推导，或提供相容的一参数调用方式。这样才能展示后续方法如何从“算法逻辑大致正确但 API 不相容”的错误程序，一步步修到 public test 可通过。

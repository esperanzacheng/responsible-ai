# Role-Sensitive Ethical Risk Analysis

## 文件說明

### 1. 數據文件
- `AsiaCement/examples.jsonl` - 亞洲水泥案例數據（已創建）
- 每行包含一個 JSON 對象，格式如下：

```json
{
  "role": "角色名稱",
  "question": "問題",
  "harmful_response": "高風險回應",
  "safety_anchored_response": "安全錨定回應",
  "type": "類型",
  "risk_category": "風險類別"
}
```

### 2. 分析腳本
- `analyze_role_safety.py` - 主要分析工具

## 使用方法

### 基本使用

```bash
cd "/Users/cheng/Documents/Mannheim/2025_Fall/Responsible AI/code"
python analyze_role_safety.py
```

### 功能說明

1. **顯示數據統計** - 查看角色分佈和風險類別
2. **查看案例詳情** - 查看特定案例的完整分析
3. **生成訓練數據** - 生成帶有 Chain-of-Thought 的訓練數據
4. **退出** - 結束程序

### Chain-of-Thought 訓練數據

生成的訓練數據包含：
- **Role Context** - 角色背景和價值觀
- **Ethical Risk Assessment** - 倫理風險評估
- **Response Comparison** - 對比高風險和安全回應
- **Critical Reflection** - 批判性反思

### 示例輸出

```
Chain-of-Thought Analysis:

Step 1 - Role Context:
What are the institutional objectives and values of this role?

Step 2 - Ethical Risk Assessment:
What ethical risks might arise if role fidelity is prioritized over ethical constraints?
Risk Category: power_asymmetry_justification

Step 3 - Compare Responses:
[Harmful Response vs Safety-Anchored Response]

Step 4 - Critical Reflection:
What assumptions does the harmful response normalize?
How does the safety-anchored response maintain role relevance while addressing ethical concerns?
```

## 風險類別

1. **power_asymmetry_justification** - 權力不對稱的合理化
2. **marginalization_of_consent** - 邊緣化同意權
3. **procedural_deflection** - 程序性迴避

## 擴展方法

### 添加新案例

在 `AsiaCement/examples.jsonl` 中添加新行：

```json
{"role": "角色", "question": "問題", "harmful_response": "...", "safety_anchored_response": "...", "type": "...", "risk_category": "..."}
```

### 自定義分析

修改 `analyze_role_safety.py` 中的 `generate_chain_of_thought_prompt()` 方法來自定義 prompt 格式。

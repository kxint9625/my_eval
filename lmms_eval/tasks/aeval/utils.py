import re
from pathlib import Path
from typing import List, Dict
from PIL import Image

# =======================
# 答案归一化 & 评估函数
# =======================
def normalize_answer(text: str) -> str:
    """统一答案格式: yes/no, A/B/C/D, 数字"""
    if text is None:
        return ""

    text = str(text).strip().lower()

    # 去掉无关符号
    text = text.replace("answer", "")
    text = re.sub(r"[^a-z0-9.\-]", " ", text)  # 保留字母数字小数点负号
    text = text.strip()

    # yes/no 标准化
    if text.startswith("y"):
        return "yes"
    if text.startswith("n"):
        return "no"

    # 单字母选项 (a/b/c/d/e)
    if len(text) == 1 and text in ["a", "b", "c", "d", "e"]:
        return text

    # 括号形式 (a)
    m = re.match(r"\(?([a-e])\)?", text)
    if m:
        return m.group(1)

    # 数字
    if re.match(r"^-?\d+(\.\d+)?$", text):
        return str(float(text)) if "." in text else str(int(text))

    return text


# def closed_form_acc(answer: str, pred: str) -> int:
#     """精确匹配 yes/no, a/b/c/d, 数字"""
#     a = normalize_answer(answer)
#     p = normalize_answer(pred)
#     return 1 if a == p and a != "" else 0

def closed_form_acc(answer: str, pred: str) -> int:
    """数值题 ±25% 容忍；否则精确匹配"""
    a = normalize_answer(answer)
    p = normalize_answer(pred)

    # 尝试提取数字
    def extract_number(s: str):
        match = re.search(r"-?\d+(\.\d+)?", s)
        return float(match.group()) if match else None

    gt_num = extract_number(a)
    pred_num = extract_number(p)

    # 如果两边都是数字，用 ±25% 容忍度
    if gt_num is not None and pred_num is not None:
        lower, upper = gt_num * 0.75, gt_num * 1.25
        return 1 if lower <= pred_num <= upper else 0

    # 否则回退到精确匹配
    return 1 if a == p and a != "" else 0


# =======================
# NuScenes 适配函数
# =======================
def nuscenes_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """从 all_camera_image_paths 取出相机图像"""
    visuals = []
    for _, img_path in doc.get("all_camera_image_paths", {}).items():
        try:
            visuals.append(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[WARN] 打不开 {img_path}: {e}")
            continue
    return visuals


def nuscenes_doc_to_text(doc: Dict, lmms_eval_specific_kwargs=None) -> str:
    """返回 question 作为输入 prompt"""
    return doc["question"]


def nuscenes_doc_to_target(doc: Dict) -> str:
    """返回 ground-truth 答案"""
    return doc["answer"]


def nuscenes_process_results(doc: Dict, results: List[str]) -> Dict:
    """对比预测和答案，返回一条记录"""
    pred = results[0] if results else ""
    gt = doc["answer"]

    score = closed_form_acc(gt, pred)

    model_response = {
        "sample_id": doc.get("sample_token", None),
        "sub_task": doc.get("sub_task", "nuscenes"),
        "question_type": "closed-form",
        "answer": gt,
        "parsed_pred": pred,
        "score": score,
    }

    return {"overall_score": model_response}


# =======================
# 聚合函数
# =======================
def overall_score(results: List[Dict]) -> float:
    """计算所有样本的平均分"""
    if not results:
        return 0.0
    total, count = 0.0, 0
    for r in results:
        if "score" in r:
            total += r["score"]
            count += 1
        elif "overall_score" in r and "score" in r["overall_score"]:
            total += r["overall_score"]["score"]
            count += 1
    return total / count if count > 0 else 0.0

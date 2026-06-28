import json
import os
from typing import Dict, List

import requests
from dotenv import load_dotenv


class StrategyExplainer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BAILI_API_KEY", "").strip()
        self.api_url = os.getenv("BAILI_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
        self.model = os.getenv("BAILI_MODEL", "qwen-plus").strip()

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_url and self.model)

    def _fallback_single(self, row: Dict) -> str:
        coupon = row.get("coupon") or "不发券"
        segment = row.get("segment", "未知分层")
        member = row.get("会员状态", "未知会员")
        lifecycle = row.get("用户生命周期阶段", "未知阶段")
        beta = float(row.get("beta", 0.0))
        delta = float(row.get("delta", 0.0))
        roi = float(row.get("roi", 0.0))
        value = float(row.get("V", 0.0))
        sensitivity = float(row.get("S_mean", 0.0))

        if coupon == "不发券":
            return (
                f"{row.get('user_id')} 当前建议不发券。该用户属于{segment}，自然转化率约为 {beta:.2f}，"
                f"券带来的增量有限，预计 ROI 仅为 {roi:.2f}，因此更适合保留预算给更高响应人群。"
            )

        return (
            f"{row.get('user_id')} 推荐发放 {coupon}。该用户属于{segment}，会员层级为 {member}，"
            f"处于{lifecycle}阶段，用户价值约 {value:.1f}，价格敏感度约 {sensitivity:.2f}。"
            f"在当前模型下，自然转化率约 {beta:.2f}，该券可带来约 {delta:.3f} 的购买阶段增量，"
            f"预计 ROI 为 {roi:.2f}，因此具备较高补贴效率。"
        )

    def explain_batch(self, records: List[Dict], business_context: Dict = None) -> List[Dict]:
        if not records:
            return []

        if not self.is_available():
            prefix = ""
            if business_context and business_context.get("impact_text"):
                prefix = f"{business_context['impact_text']}。"
            return [{"user_id": row.get("user_id"), "explanation": prefix + self._fallback_single(row)} for row in records]

        payload_rows = []
        for row in records:
            payload_rows.append({
                "user_id": row.get("user_id"),
                "segment": row.get("segment"),
                "member": row.get("会员状态"),
                "lifecycle": row.get("用户生命周期阶段"),
                "payment_band": row.get("历史实付分层（365天）"),
                "coupon": row.get("coupon"),
                "V": round(float(row.get("V", 0.0)), 3),
                "A": round(float(row.get("A", 0.0)), 3),
                "S_mean": round(float(row.get("S_mean", 0.0)), 3),
                "beta": round(float(row.get("beta", 0.0)), 4),
                "delta": round(float(row.get("delta", 0.0)), 4),
                "delta_gtv_causal": round(float(row.get("delta_gtv_causal", 0.0)), 3),
                "roi": round(float(row.get("roi", 0.0)), 3),
            })

        business_context = business_context or {}
        impact_text = business_context.get("impact_text", "")
        prompt = (
            "你是一名电商/本地生活平台的业务策略分析师。"
            "请根据给定的用户状态和模型输出，为每个用户生成1段面向业务同学的中文策略解释。"
            "要求：1）说明为什么发这个券或为什么不发券；2）语言业务化、易懂；3）每条 60~100 字；"
            "4）避免空话，必须结合用户价值、价格敏感度、自然转化率和 ROI；"
            "5）必须输出商业价值表达（用“每天X次发券/每月”量级换算），不要只报 ROI 数字；"
            f"6）商业价值换算口径：{impact_text if impact_text else '若缺失则你自行用“每天1万次发券、按30天”给出近似换算'}；"
            "5）输出严格为 JSON 数组，每个元素包含 user_id 和 explanation 两个字段。\n"
            f"输入数据：{json.dumps(payload_rows, ensure_ascii=False)}"
        )

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你输出严格 JSON，不要加 markdown。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4
                },
                timeout=45
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.replace("json", "", 1).strip()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        return [{"user_id": row.get("user_id"), "explanation": self._fallback_single(row)} for row in records]

"""Generate invented Excel inputs for a public, reproducible demo.

No private data is read or transformed. Output goes to the ignored data/raw/
directory and must never be described as competition observations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(__file__).resolve().parent / "data" / "raw"
FILES = ("用户画像表.xlsx", "神券订单数据样例.xlsx", "用户行为序列.xlsx")


def generate(output_dir=OUTPUT, seed=42):
    output_dir = Path(output_dir)
    existing = [name for name in FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace existing inputs: {', '.join(existing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    user_ids = [f"DEMO{i:04d}" for i in range(1, 81)]

    profiles = pd.DataFrame({
        "user_id": user_ids,
        "常驻城市": rng.choice(["北京", "上海", "杭州"], len(user_ids)),
        "年龄": rng.integers(20, 51, len(user_ids)),
        "用户生命周期阶段": rng.choice(["新客", "老客", "流失召回"], len(user_ids)),
        "会员状态": rng.choice(["L1", "L2", "L3", "L4", "L5"], len(user_ids)),
        "历史实付分层（365天）": rng.choice(["低客单价", "中客单价", "高客单价"], len(user_ids)),
    })

    orders = []
    for index in range(240):
        amount = float(np.clip(rng.lognormal(np.log(28), 0.35), 8, 90))
        subsidy = int(rng.choice([0, 0, 0, 3, 5, 8]))
        orders.append({
            "user_id": user_ids[index % len(user_ids)],
            "下单时间": pd.Timestamp("2026-01-01 14:00") + pd.Timedelta(minutes=30 * (index % 6)),
            "BU": "饮品", "POI分类": "奶茶", "订单金额": round(amount, 2),
            "美补金额": subsidy,
            "1-免费神券，2-付费神券": 1 if index % 3 else 2,
            "1-非膨胀券，2-膨胀券": 1 if index % 4 else 2,
        })

    behaviors = []
    for index, user_id in enumerate(user_ids):
        stages = ["搜索"]
        if index % 5:
            stages.append("点击店铺")
        if index % 3:
            stages.append("加入购物车")
        if index % 4:
            stages.append("下单")
        for minute, stage in enumerate(stages):
            behaviors.append({
                "用户id": user_id, "session_id": f"S{index:04d}",
                "行为时间戳": f"T-1 15:{minute:02d}",
                "行为类型": stage, "具体内容": "奶茶",
            })

    for name, frame in zip(FILES, (profiles, pd.DataFrame(orders), pd.DataFrame(behaviors))):
        frame.to_excel(output_dir / name, index=False)
    return {name: len(frame) for name, frame in zip(FILES, (profiles, orders, behaviors))}


if __name__ == "__main__":
    for name, count in generate().items():
        print(f"Synthetic demo: {name} ({count} rows)")

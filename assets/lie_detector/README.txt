测谎 / 谎言探测仪
================

路径：
  assets/lie_detector/lie.png           英文 LIE DETECTOR 标题
  assets/lie_detector/lie_en.png        同上备份
  assets/lie_detector/lie_cn_title.png  盛趣「谎言探测仪」标题
  assets/lie_detector/shapes/           ★ 任意图形裁图目录（不限星形）
      star.png / star_faded.png         已有示例；可再加 heart.png、diamond.png…
  assets/lie_detector/ref_cn_dialog.jpg 盛趣弹窗参考截图

检测：
  - 模板匹配标题栏 + OCR 关键词
  - 命中后：响铃、暂停挂机
  - 默认自动录像：触发前约 2 秒 + 整段测谎 → _debug_out/lie_clips/lie_时间戳.mp4
    关闭：lie_alert.record_clip: false

鼠标跟随（lie_alert.auto_follow）——图形随机：
  目标是「相对周围明显更亮的白色几何块」（星/心/菱形…），会移动并渐隐。
  策略（防跟错）：
    1) 说明文字/倒计时还在时先不跟
    2) 用局部对比度找白色目标，丢掉岩石高光、HUD、大号倒计时
    3) 锁定后只在附近小范围搜索，禁止远距离跳到别的亮块
    4) shapes/ 模板仅在相关度很高时才用
  调试窗：红圈 + 绿十字。

补充其它图形模板（可选，更稳）：
  1. 小游戏里截一张清晰实心白图
  2. 只裁图形本身 → assets/lie_detector/shapes/随便命名.png
  3. 重启脚本即可自动加载

关闭跟随（只响铃）：
  lie_alert.auto_follow: false

注意：跟随会移动系统鼠标；测谎时请勿同时手动抢鼠标。

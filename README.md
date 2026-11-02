# 前言
* 本项目参考了这个库[MapleStoryAutoLevelUp](https://github.com/KenYu910645/MapleStoryAutoLevelUp) 
* 只要是市面上的079版本的冒险岛都可以使用，测试过的版本有：wingms, mapleStoryGo。
* 还有，开挂是不对的，本人已经被封了3个号。

# 功能
 * [x] mini地图识别 (还有问题，需要调试)
 * [x] 角色识别 
 * [x] 怪物识别
 * [ ] 地图绘制 
 * [x] 自动打怪
 * [x] 自动吃药
 * [x] 测谎报警 + 谎言探测仪跟随（白形粘滞，褪色后跟旋转目标；绿框只做干扰，auto_follow 可关）
 * [x] F1 / F2 / F3 多模式挂机（启动空闲，按键切换）

# F1 / F2 / F3 挂机逻辑与按键

配置见 `config/config_custom.yaml` → `profiles:`。启动后默认**空闲**，按对应功能键加载配置并开始，**再按同键关闭**。

## 公共按键（`key:`）

| 按键 | 作用 |
|------|------|
| ← / → | 左右移动 / 巡逻换向 |
| ↑ / ↓ | 上下（F2 在平台偏高时会按 ↓） |
| `A` | 群攻（`aoe_skill`） |
| `Z` | 闪现（仅 F1 开启） |
| `End` | HP 药水（全局）；F2 另作养宠物 |
| `Home` | MP 药水 |
| `F1` / `F2` / `F3` | 切换对应挂机配置 |
| `F11` | 检测/猎杀相关开关（日志提示） |

测谎弹出时会暂停挂机；`auto_follow=true` 时会自动跟形（需 bot 与游戏同为管理员权限，鼠标才能动）。

---

## F1 — 甩锅侠12139 · 全图黄点边缘巡逻

**定位**：名字牌 `character/name.png`（甩锅侠12139）

**挂机逻辑**
- `flash_patrol`：左右巡逻
- 用小地图**黄点 X** 判断是否到地图左右边缘（约 6%）再换向
- 边缘卡住才按「顶墙」逻辑换向；中区不靠怪物左右数转向
- **不开**怪物模板识别

**攻击**
- 只按 `A`，间隔约 **0.5–0.75s**
- **开 Z 闪现**，约每 7.5–8.5s 闪一次（左右移动时）

**Buff**（施法会停顿移动/攻击约 2.5s；开挂先加完再开战）
| 键 | 间隔 |
|----|------|
| `W` | 2 分钟 |
| `Shift` | 5 分钟 |
| `End` | 25 分钟 |

---

## F2 — 甩锅侠12138 · 云彩公园中区巡逻（跳 A）

**定位**：名字牌 `character/name_12138.png`（甩锅侠12138）

**挂机逻辑**
- `flash_patrol`：左右巡逻，但只在小地图黄点 **约 10%～80%** 中区来回（到界立刻掉头）
- 黄点 Y 偏高（平台）时按住 `↓`
- **不开 Z**（本图不能闪现）
- **不开**怪物识别 / 怪物转向
- **不加战斗 Buff**，开挂即打

**攻击**
- 连招 `S` → 0.1s → `A`（跳起空中 A）
- 每轮间隔 **0.8s**

**养宠物**（无施法动作，**不停顿**）
| 键 | 间隔 |
|----|------|
| `End` | 15 分钟（开 F2 约 2s 先喂一次） |

---

## F3 — 甩锅侠12139 · 通天塔定点站桩

**定位**：名字牌 `character/name.png`（甩锅侠12139）

**挂机逻辑**
- **定点站桩**：开启瞬间锁定小地图黄点为中心
- 偏离中心超过约 **6px** 才走回，不左右扫图
- **不开**闪现、**不开**怪物识别

**攻击**
- 只按 `A`，间隔约 **1.5–2.0s**

**Buff**（与 F1 相同：`W` 2分 / `Shift` 5分 / `End` 25分，施法停顿约 2.5s，先加完再开战）

---

## 对照一览

| | F1 | F2 | F3 |
|--|----|----|-----|
| 角色 | 12139 | 12138 | 12139 |
| 移动 | 全图边缘巡逻 | 中区 10%～80% | 定点站桩 |
| 攻击 | `A` 0.5–0.75s | `S`+`A` 0.8s | `A` 1.5–2.0s |
| `Z` 闪现 | 开 | 关 | 关 |
| Buff | W/Shift/End | 无 | W/Shift/End |
| 养宠物 | 无 | End 每 15 分 | 无 |

# 1、识别角色 + 怪物 + miniMap
## 1-1、演示
![识别怪物演示](demo_gif/recognize_user_map_monster.gif)
## 1-2、素材库找到怪物素材
* [素材下载地址](https://maplestory.wiki/GMS/83/mob/1110101)
* 把走动的素材下载下载，放到`monster/下载的怪物名称`文件夹下
* 大概长这样![下载的怪物素材demo](demo_gif/monster.png)
* 如何调整配置
  * 打开`config/config_data.yaml`，找到`mob`字段
  * 添加新的怪物配置，示例如下：只要改这三个字段就行
```yaml
monsters: ["black_axe_stump"]  # 指定怪物文件夹，数组可多个
diff_thres: 0.46               # 匹配阈值（越小越严格）
scales: [1]                    # 缩放比例，官方素材不匹配时改: [0.7, 0.85, 1.0, 1.15]            # 每N帧检测一次 (1=每帧, 3=流畅优先)
  ```

# 2、地图绘制
TODO

# 3、单独测试称号定位

如果你只想验证 `character/title.png` 能不能定位到角色，不想直接跑完整挂机流程，可以先跑：

```powershell
python .\test\test_title_locator.py
```

默认会读取：

- `config/config_default.yaml`
- `config/config_custom.yaml`
- `test/test_image/screenshot.png`

并输出：

- `score`
- 是否命中 `hit=True/False`
- `title_loc`
- `player_loc`

同时会生成调试图：

- `screenshot/title_locator_debug.png`

如果要换测试图：

```powershell
python .\test\test_title_locator.py --image test\test_image\screenshot.png
```

## Star History

<a href="https://www.star-history.com/?repos=zhoufanglu%2Fmaplestory-auto-ds&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zhoufanglu/maplestory-auto-ds&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zhoufanglu/maplestory-auto-ds&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=zhoufanglu/maplestory-auto-ds&type=date&legend=top-left" />
 </picture>
</a>

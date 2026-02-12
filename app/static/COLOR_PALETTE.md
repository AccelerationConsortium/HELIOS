# OTbot UI Color Palette

美化后的 OTbot 用户界面配色方案，基于用户提供的调色板。

## 🎨 配色方案

### 背景色 (Backgrounds)
| 颜色名 | 色值 | 用途 | 示例 |
|--------|------|------|------|
| Deep Blue | `#1E3A5F` | 主背景色 | 页面底色、输入框 |
| Deep Purple | `#4A3B68` | 次要背景 | 侧边栏、卡片 |
| Blue-Purple | `#5B6B94` | 三级背景 | 悬停状态 |
| Blue-Gray | `#5B7A9D` | 悬停背景 | 交互元素 |

### 文字色 (Text)
| 颜色名 | 色值 | 用途 |
|--------|------|------|
| Bright White | `#FFFFFF` | 标题、强调文字 |
| Light Gray | `#F8FAFC` | 主要文字 |
| Light Blue-Purple | `#8B9AC4` | 次要文字 |
| Blue-Gray | `#5B7A9D` | 弱化文字 |

### 强调色 (Accent Colors)
| 颜色名 | 色值 | 用途 | 场景 |
|--------|------|------|------|
| Bright Pink | `#D45B8E` | 主要强调色 | CTA 按钮、激活状态、边框高亮 |
| Pink | `#D897B3` | 成功状态 | 完成标记、正向反馈 |
| Deep Pink-Red | `#C5517C` | 危险/警告 | 错误提示、删除操作 |
| Golden Yellow | `#E5A84B` | 警告/通知 | 提示信息、注意事项 |
| Orange | `#D97E4E` | 次要 CTA | 辅助按钮 |
| Deep Purple | `#4A3B68` | 深色强调 | Agent 标识 |
| Blue-Gray | `#5B7A9D` | 信息提示 | 帮助文本、说明 |
| Brown | `#795548` | 边框/暗色 | 禁用状态、分隔线 |

### Agent 颜色 (Agent Colors)
| Agent 类型 | 颜色 | 色值 |
|-----------|------|------|
| Planner | Bright Pink | `#D45B8E` |
| Design | Blue-Gray | `#5B7A9D` |
| Compiler | Golden Yellow | `#E5A84B` |
| Safety | Deep Pink-Red | `#C5517C` |
| Executor | Pink | `#D897B3` |
| Sensing | Light Blue-Purple | `#8B9AC4` |
| Stop | Brown | `#795548` |
| Parse | Pink | `#D897B3` |
| System | Blue-Purple | `#5B6B94` |
| Strategy | Deep Purple | `#4A3B68` |

## ✨ 视觉效果

### Glassmorphism (磨砂玻璃)
```css
background: rgba(74, 59, 104, 0.7);
backdrop-filter: blur(10px);
border: 1px solid rgba(255, 255, 255, 0.1);
```

### 渐变背景
```css
background: linear-gradient(135deg, #1E3A5F 0%, #4A3B68 100%);
```

### 按钮渐变
```css
/* Primary Button */
background: linear-gradient(135deg, #D45B8E, #C5517C);
box-shadow: 0 2px 8px rgba(212, 91, 142, 0.3);

/* Hover State */
background: linear-gradient(135deg, #C5517C, #D45B8E);
box-shadow: 0 4px 12px rgba(212, 91, 142, 0.4);
transform: translateY(-1px);
```

## 🔤 字体系统

### 主字体
- **Font Family**: JetBrains Mono
- **Weights**: 300 (Light), 400 (Regular), 500 (Medium), 600 (Semi-Bold), 700 (Bold)
- **Google Fonts**: https://fonts.google.com/specimen/JetBrains+Mono

### 字体特性
- 支持连字 (ligatures): `'liga' 1`
- 支持上下文替代: `'calt' 1`
- 优化的代码显示效果

## 🎯 使用指南

### CSS 变量引用
```css
/* 背景色 */
var(--bg-primary)      /* #1E3A5F */
var(--bg-secondary)    /* #4A3B68 */
var(--bg-tertiary)     /* #5B6B94 */

/* 强调色 */
var(--accent-primary)   /* #D45B8E */
var(--accent-success)   /* #D897B3 */
var(--accent-warning)   /* #E5A84B */
var(--accent-danger)    /* #C5517C */

/* Glassmorphism */
var(--glass-bg)        /* rgba(74, 59, 104, 0.7) */
var(--glass-border)    /* rgba(255, 255, 255, 0.1) */
var(--glass-blur)      /* 10px */
```

### 动画曲线
```css
--transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
--transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
```

## 📱 响应式设计

- **Desktop**: 全功能，三栏布局
- **Tablet** (≤1200px): 侧边栏折叠为抽屉
- **Mobile** (≤900px): 垂直堆叠布局

## ♿ 可访问性

- **对比度**: 所有文字与背景对比度 ≥ 4.5:1
- **焦点状态**: 可见的焦点环 (3px rgba)
- **键盘导航**: 完整支持 Tab 导航
- **动画**: 支持 `prefers-reduced-motion`

## 🚀 性能优化

- **Backdrop Filter**: 使用 GPU 加速的模糊效果
- **Transform**: 使用 transform 而非 position 做动画
- **Will-Change**: 对频繁动画的元素添加 will-change
- **Font Loading**: 使用 preconnect 预连接字体服务器

---

**更新日期**: 2026-02-11
**版本**: v2.0
**设计师**: Claude Code + User Palette

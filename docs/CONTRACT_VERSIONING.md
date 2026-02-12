# Contract Versioning System

## 概述

OTbot的Contract Versioning System提供了**自动schema migration**和**向后兼容性保证**，让contracts可以安全演进而不破坏现有数据。

## 核心特性

### ⭐⭐⭐⭐ 级别特性（已实现）

1. **Semantic Versioning** - 所有contracts使用`major.minor.patch`版本号
2. **Automatic Migration** - 旧版本数据自动升级到当前schema
3. **Migration Path Discovery** - BFS算法找到最短升级路径
4. **Checksum Validation** - SHA256校验数据完整性
5. **Backward Compatibility** - 旧代码仍能正常工作
6. **Type Safety** - Pydantic强类型验证 + 类型提示
7. **Audit Trail** - `migrated_from`字段记录升级历史

### 🎯 下一步优化（⭐⭐⭐⭐⭐）

- [ ] Formal verification with Z3/SMT solver
- [ ] W3C PROV-O provenance tracking
- [ ] Contract registry with OpenAPI spec generation
- [ ] Cross-platform adoption (Hamilton, Tecan等)

---

## 快速开始

### 1. 定义新Contract

```python
from app.contracts.versioning import BaseVersionedContract
from typing import ClassVar

class MyContract(BaseVersionedContract):
    # 版本元数据（必须）
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    CONTRACT_NAME: ClassVar[str] = "MyContract"

    # 业务字段
    field1: str
    field2: int
```

### 2. 注册Migration

创建`app/contracts/migrations/my_contract_migrations.py`：

```python
from app.contracts.versioning import register_migration

@register_migration("MyContract", from_version="1.0.0", to_version="2.0.0")
def migrate_my_contract_v1_to_v2(data: dict) -> dict:
    """v1.0 → v2.0: 添加新字段field3"""
    # 添加新字段（带默认值）
    data.setdefault("field3", [])

    # 重命名字段
    if "old_name" in data:
        data["new_name"] = data.pop("old_name")

    # 删除过时字段
    data.pop("obsolete_field", None)

    return data
```

### 3. 导入Migrations

在`app/contracts/migrations/__init__.py`添加：

```python
from app.contracts.migrations import my_contract_migrations
```

### 4. 使用Auto-Migration

```python
# 加载v1.0.0数据（模拟从旧DB读取）
old_data = {
    "schema_version": "1.0.0",
    "field1": "value",
    "field2": 42,
}

# 自动升级到v2.0.0
contract = MyContract.from_dict(old_data)

assert contract.schema_version == "2.0.0"
assert contract.migrated_from == "1.0.0"
assert contract.field3 == []  # 新字段已添加
```

---

## 实战案例：TaskContract v1.0 → v2.0

### 版本历史

| Version | Changes | Date |
|---------|---------|------|
| **1.0.0** | 初始版本，使用`version`字段 | 2025-01 |
| **2.0.0** | 重命名`version`→`schema_version`<br>添加`protocol_metadata`<br>添加`deprecation_warnings` | 2026-02 |

### Migration代码

```python
@register_migration("TaskContract", from_version="1.0.0", to_version="2.0.0")
def migrate_task_contract_v1_to_v2(data: dict) -> dict:
    # 1. 重命名字段（向后兼容）
    if "version" in data and "schema_version" not in data:
        data["schema_version"] = data.pop("version")

    # 2. 添加新字段
    data.setdefault("protocol_metadata", {})
    data.setdefault("deprecation_warnings", [])

    return data
```

### 使用示例

```python
# 旧数据库记录（v1.0.0）
v1_task = {
    "contract_id": "tc-abc123",
    "version": "1.0",  # 旧字段名
    "objective": {...},
    "exploration_space": {...},
    # ...没有protocol_metadata...
}

# 自动升级
task = TaskContract.from_dict(v1_task)

# 验证
assert task.schema_version == "2.0.0"
assert task.migrated_from == "1.0.0"
assert task.protocol_metadata == {}  # 已添加
assert hasattr(task, 'deprecation_warnings')  # 已添加

# 数据完整性校验
assert task.verify_checksum()
```

---

## 高级特性

### 多步Migration

系统自动处理多步升级路径：

```python
# 注册migration链
@register_migration("MyContract", "1.0.0", "2.0.0")
def v1_to_v2(data): ...

@register_migration("MyContract", "2.0.0", "3.0.0")
def v2_to_v3(data): ...

# 自动找到路径：1.0 → 2.0 → 3.0
old_data = {"schema_version": "1.0.0", ...}
contract = MyContract.from_dict(old_data)  # 自动2步升级
assert contract.schema_version == "3.0.0"
assert contract.migrated_from == "1.0.0"
```

### Checksum验证

```python
contract = TaskContract(**data)

# 自动计算checksum
assert contract.checksum  # "a3b4c5d6..."

# 验证数据未被篡改
assert contract.verify_checksum()

# 修改后checksum不匹配
contract.protocol_pattern_id = "new_pattern"
assert not contract.verify_checksum()  # False
```

### 版本支持检查

```python
# 检查是否有migration路径
if TaskContract.supports_version("0.9.0"):
    contract = TaskContract.from_dict(old_data)
else:
    raise ValueError("Unsupported schema version")
```

---

## Migration最佳实践

### ✅ DO

1. **总是添加默认值**
   ```python
   data.setdefault("new_field", [])  # ✅ Good
   ```

2. **向后兼容重命名**
   ```python
   if "old_name" in data:
       data["new_name"] = data.pop("old_name")
   ```

3. **清理过时字段**
   ```python
   data.pop("obsolete_field", None)  # 静默删除
   ```

4. **保留数据类型**
   ```python
   # ✅ Good: 类型不变
   data["count"] = int(data.get("count", 0))

   # ❌ Bad: 改变类型可能破坏代码
   data["count"] = str(data.get("count", 0))
   ```

5. **文档化Breaking Changes**
   ```python
   @register_migration("MyContract", "2.0.0", "3.0.0")
   def v2_to_v3(data):
       """v2→v3: BREAKING - removed support for deprecated API."""
       if "deprecated_api_key" in data:
           data["deprecation_warnings"].append(
               "deprecated_api_key removed in v3.0.0"
           )
       return data
   ```

### ❌ DON'T

1. **不要丢失数据**
   ```python
   data.pop("important_field")  # ❌ 永久丢失！
   ```

2. **不要假设字段存在**
   ```python
   data["new"] = data["old"].upper()  # ❌ 如果old不存在会崩溃
   data["new"] = data.get("old", "").upper()  # ✅ 安全
   ```

3. **不要修改语义**
   ```python
   # ❌ Bad: "temp"含义改变
   data["temp_celsius"] = data.pop("temp")  # temp原来是华氏度？
   ```

---

## 测试Migration

```python
def test_my_contract_migration():
    """测试v1→v2 migration."""
    # 1. 准备v1数据
    v1_data = {
        "schema_version": "1.0.0",
        "field1": "value",
        "field2": 42,
    }

    # 2. 执行migration
    contract = MyContract.from_dict(v1_data)

    # 3. 验证升级
    assert contract.schema_version == "2.0.0"
    assert contract.migrated_from == "1.0.0"

    # 4. 验证新字段
    assert contract.field3 == []

    # 5. 验证旧数据保留
    assert contract.field1 == "value"
    assert contract.field2 == 42

    # 6. 验证checksum
    assert contract.verify_checksum()
```

---

## 性能考虑

- **Migration开销**: 每个migration步骤 < 1ms
- **Checksum计算**: SHA256 + JSON序列化 < 5ms
- **内存**: 只复制dict，不复制整个contract对象
- **缓存**: 可扩展添加migration结果缓存

---

## 故障排查

### Migration失败

```python
try:
    contract = TaskContract.from_dict(old_data)
except MigrationError as e:
    print(f"Migration failed: {e}")
    # 检查migration path
    path = get_migration_path("TaskContract", "1.0.0", "2.0.0")
    print(f"Expected path: {path}")
```

### Checksum不匹配

```python
if not contract.verify_checksum():
    print(f"Stored: {contract.checksum}")
    print(f"Computed: {contract.compute_checksum()}")
    # 重新计算并保存
    contract.checksum = contract.compute_checksum()
```

---

## 路线图

### Phase 1: 核心框架 ✅ (完成)
- BaseVersionedContract
- Migration registry
- Automatic migration
- Checksum validation
- Tests

### Phase 2: 全面推广 (进行中)
- [ ] 更新所有contracts继承BaseVersionedContract
- [ ] 为CampaignPlan, RunBundle, ResultPacket添加migrations
- [ ] 集成到API endpoints (读取时自动migration)

### Phase 3: 高级特性
- [ ] W3C PROV-O provenance chain
- [ ] Formal verification (Z3)
- [ ] OpenAPI schema generation
- [ ] Migration dry-run mode
- [ ] Contract diff tool

### Phase 4: 生态系统
- [ ] 发布论文/技术博客
- [ ] 开源contract specification
- [ ] 跨平台adoption (Hamilton, Tecan)
- [ ] 成为行业标准

---

## 参考

- **Semantic Versioning**: https://semver.org/
- **W3C PROV-O**: https://www.w3.org/TR/prov-o/
- **Pydantic**: https://docs.pydantic.dev/
- **Database Migration**: Alembic, Flyway concepts

---

**Status**: ⭐⭐⭐⭐ (Production-Ready)
**Next Target**: ⭐⭐⭐⭐⭐ (Industry Standard)
**Version**: 1.0.0
**Last Updated**: 2026-02-11

#!/usr/bin/env python3
"""只读 T3 会话，将白名单统计字段导出到 CC Switch 原有的 agy 扫描入口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys


def varint(data, offset=0):
    value = shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def fields(data):
    result, offset = {}, 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        wire = tag & 7
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            if len(value) != size:
                raise ValueError("truncated protobuf")
            offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError("unsupported protobuf wire type")
        result[tag >> 3] = value
    return result


def encode(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def blob(tag, data):
    return encode(tag * 8 + 2) + encode(len(data)) + data


def generation(db, step):
    """只保留模型、两个 Duration 和关联编号，不复制原始响应 metadata。"""
    row = db.execute("SELECT metadata FROM steps WHERE idx=? AND step_type=15", (step,)).fetchone()
    if not row or row[0] is None:
        return None
    reference = fields(row[0]).get(20)
    if reference is None:
        return None
    index = fields(reference).get(3, 0)
    row = db.execute("SELECT data FROM gen_metadata WHERE idx=?", (index,)).fetchone()
    if not row:
        return None
    response = fields(fields(row[0]).get(1, b""))
    clean = b""
    for tag in (11, 12):
        if isinstance(response.get(tag), bytes):
            duration = fields(response[tag])
            clean += blob(tag, b"".join(encode(k * 8) + encode(duration[k])
                                       for k in (1, 2) if isinstance(duration.get(k), int)))
    for tag in (19, 21):
        if isinstance(response.get(tag), bytes):
            clean += blob(tag, response[tag])
    return blob(20, encode(3 * 8) + encode(index)), index, blob(1, clean)


def export_session(transcript, target):
    rows = {}
    for line in transcript.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # agy 的最后一行可能尚未写完。
        if isinstance(row, dict) and isinstance(row.get("step_index"), int):
            rows[row["step_index"]] = row
    session = transcript.parents[2].name
    root = transcript.parents[4]
    source_db = root / "conversations" / (session + ".db")
    records, native = [], []
    # 保留 WAL 的一致性快照；数据库暂忙时失败重试，不用 immutable 忽略新记录。
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True, timeout=0.1) as db:
        db.execute("BEGIN")
        for step, row in sorted(rows.items()):
            if row.get("type") != "PLANNER_RESPONSE" or row.get("status") != "DONE":
                continue
            record = {"step_index": step, "type": "PLANNER_RESPONSE", "status": "DONE"}
            for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                value = row.get(key, 0)
                if not isinstance(value, int) or value < 0:
                    raise ValueError("invalid token counter")
                record[key] = value
            if isinstance(row.get("created_at"), str):
                record["created_at"] = row["created_at"]
            data = generation(db, step)
            if data:
                metadata, index, response = data
                native.append((step, metadata, index, response))
                models = fields(fields(response)[1])
                model = models.get(21) or models.get(19)
                if model:
                    record["model"] = model.decode()
            records.append(record)
    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records)
    signature = hashlib.sha256(content.encode() + b"".join(
        str(step).encode() + metadata + str(index).encode() + response
        for step, metadata, index, response in native)).hexdigest()
    destination = target / "brain" / session / ".system_generated/logs/transcript_full.jsonl"
    marker = destination.with_suffix(".sha256")
    if marker.exists() and marker.read_text() == signature:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    (target / "conversations").mkdir(exist_ok=True)
    with sqlite3.connect(target / "conversations" / (session + ".db")) as db:
        db.executescript("CREATE TABLE IF NOT EXISTS steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB);"
                         "CREATE TABLE IF NOT EXISTS gen_metadata (idx INTEGER PRIMARY KEY, data BLOB);")
        for step, metadata, index, response in native:
            db.execute("INSERT OR REPLACE INTO steps VALUES (?,15,?)", (step, metadata))
            db.execute("INSERT OR REPLACE INTO gen_metadata VALUES (?,?)", (index, response))
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(destination)
    marker.write_text(signature)
    return True


def sync(source, target):
    seen, updated, failed = set(), 0, 0
    for transcript in sorted(source.glob("*/antigravity-acp/brain/*/.system_generated/logs/transcript_full.jsonl")):
        session = transcript.parents[2].name
        if session in seen:
            continue
        seen.add(session)
        try:
            updated += export_session(transcript, target)
        except (OSError, sqlite3.Error, ValueError) as error:
            # 单个正在写入的会话不阻断其他会话；不打印正文或原始异常内容。
            print(f"T3 usage: session={session}, error={type(error).__name__}", file=sys.stderr)
            failed += 1
    if failed:
        raise RuntimeError(f"T3 usage: failed={failed}; successful updates retained; retry next run")
    return len(seen), updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".t3/userdata/providers/antigravity")
    parser.add_argument("--target", type=Path, default=Path.home() / ".cc-switch/agy/t3-usage/antigravity-cli")
    args = parser.parse_args()
    os.umask(0o077)
    count, updated = sync(args.source.resolve(), args.target.resolve())
    if updated:
        print(f"T3 usage: sessions={count}, updated={updated}")

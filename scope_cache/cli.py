"""Запуск анализа гранат на Cache."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from scope_cache import ROOT


def doctor():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    result = []
    for requirement in config["project"]["dependencies"]:
        name, expected = requirement.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        result.append(
            {"package": name, "expected": expected, "installed": actual, "ok": actual == expected}
        )
    return {
        "ok": all(row["ok"] for row in result) and sys.version_info[:2] == (3, 12),
        "python": sys.version.split()[0],
        "required_python": "3.12",
        "dependencies": result,
        "prepared_data": {
            name: (ROOT / name).is_file()
            for name in [
                "data/events.parquet",
                "data/release_context.parquet",
                "data/grenade_scores.parquet",
                "models/flash.joblib",
            ]
        },
    }


def run(source=None, retrain=False, rescore=False, refresh_diagnostics=False):
    from scope_cache.core import read_analysis, sha256_file
    from scope_cache.data import extract_events, extract_context
    from scope_cache.geometry import build_sightlines
    from scope_cache.model import train
    from scope_cache.scoring import score_dataset, player_profiles
    from scope_cache.validation import audit, diagnose_model, diagnose_smoke

    events, context = ROOT / "data/events.parquet", ROOT / "data/release_context.parquet"
    expected = "4edf4e909349d5131ac39a6d97a9bd49436772c939c7ade3d897713dca8ebe17"
    if source is not None and sha256_file(Path(source)) != expected:
        raise ValueError("Контрольная сумма исходника не соответствует датасету")
    if not events.exists():
        if source is None:
            raise FileNotFoundError("Нет таблицы бросков: укажите --source ORIGINAL.list")
        extract_events(Path(source), events, expected)
    if not context.exists():
        if source is None:
            raise FileNotFoundError("Нет контекста бросков: укажите --source ORIGINAL.list")
        extract_context(Path(source), events, context)
    if not (ROOT / "map_assets/sightlines.npz").exists():
        build_sightlines()
    model = ROOT / "models/flash.joblib"
    metadata = model.with_suffix(".metadata.json")
    stale = not model.exists() or not metadata.exists()
    if not stale:
        stale = json.loads(metadata.read_text())["source"] != {
            "events_sha256": sha256_file(events),
            "context_sha256": sha256_file(context),
        }
    if retrain or stale:
        train(events, context, model)
        refresh_diagnostics = True
    score_dataset(force=rescore)
    player_profiles()
    analysis = read_analysis()
    if (
        refresh_diagnostics
        or analysis.get("model", {}).get("model_sha256")
        != json.loads(metadata.read_text())["artifact"]["sha256"]
    ):
        diagnose_model()
    if refresh_diagnostics or "smoke" not in analysis:
        diagnose_smoke()
    result = audit(quiet=True)
    if not result["ok"]:
        raise ValueError("Проверка данных и показателей не пройдена")
    print(f"PASS: {result['counts']['checks']} проверок, {result['counts']['events']:,} бросков")


def check():
    import io

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    path = ROOT / "reports/checks.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["unit_tests"] = {
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "reports/tests.log").write_text(stream.getvalue())
    print(stream.getvalue(), end="")
    return result.wasSuccessful()


def pack(output):
    from scope_cache.core import sha256_file

    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Сохраните ZIP за пределами папки проекта")
    checks = json.loads((ROOT / "reports/checks.json").read_text())
    if not checks.get("unit_tests", {}).get("ok") or not checks.get("audit", {}).get("ok"):
        raise ValueError("Перед упаковкой выполните check и audit")
    paths = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part.startswith(".") for part in relative.parts) and not (
            relative.as_posix() == ".gitignore" or relative.parts[0] == ".github"
        ):
            continue
        if any(
            part in {"__pycache__", "venv", "figures", "demo_context"} or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if (
            path.suffix in {".zip", ".pyc", ".tmp", ".list"}
            or ".building." in path.name
            or path.name == "MANIFEST.json"
        ):
            continue
        paths.append(path)
    paths.sort()
    manifest = {
        "schema_version": 4,
        "project": "Cache Utility",
        "files": [
            {
                "path": p.relative_to(ROOT).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in paths
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="scope-package-") as tmp:
        temporary = Path(tmp) / output.name
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in paths:
                z.write(p, "Scope_Cache_PoC/" + p.relative_to(ROOT).as_posix())
            z.writestr(
                "Scope_Cache_PoC/MANIFEST.json",
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            )
        with zipfile.ZipFile(temporary) as z:
            if z.testzip():
                raise ValueError("Ошибка целостности ZIP")
            import hashlib

            for entry in manifest["files"]:
                digest = hashlib.sha256()
                with z.open("Scope_Cache_PoC/" + entry["path"]) as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Ошибка контрольной суммы: " + entry["path"])
        shutil.copyfile(temporary, output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "files": len(paths) + 1,
        "sha256": sha256_file(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Проверить окружение")
    commands.add_parser("summary", help="Показать метрики флешек и ограничения")
    commands.add_parser("audit", help="Проверить все данные и сохранённые прогнозы")
    commands.add_parser("check", help="Запустить тесты")
    commands.add_parser("diagnostics", help="Пересчитать интервалы и чувствительность смока")
    score = commands.add_parser("score", help="Оценить смок, флешку, HE или огненную гранату")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--mode", choices=["auto", "setup", "release"], default="auto")
    train = commands.add_parser("train", help="Сравнить модели и переобучить выбранную")
    train.add_argument("--without-kernel", action="store_true")
    pipeline = commands.add_parser(
        "run", help="Подготовить данные, пересчитать показатели и проверить"
    )
    pipeline.add_argument("--source", type=Path)
    pipeline.add_argument("--retrain", action="store_true")
    pipeline.add_argument("--rescore", action="store_true")
    pipeline.add_argument("--refresh-diagnostics", action="store_true")
    report = commands.add_parser("report", help="Выполнить и сохранить аналитический ноутбук")
    report.add_argument("--output", type=Path)
    demo = commands.add_parser("demo", help="Импортировать .dem или CS-NET JSON")
    demo.add_argument("--input", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--match-id", required=True)
    demo.add_argument("--expected-map", default="de_cache")
    demo.add_argument("--round-offset", type=int, default=0)
    demo.add_argument("--id-scheme", choices=["steam64", "steam-account32"], default="steam64")
    demo.add_argument("--score-smokes", action="store_true")
    demo.add_argument("--allow-geometry-mismatch", action="store_true")
    demo.add_argument("--sample-ticks", type=int, default=16)
    demo.add_argument("--join-events", action="store_true")
    mapping = commands.add_parser("map", help="Восстановить закреплённую карту или её линии обзора")
    mapping.add_argument("--fetch", action="store_true")
    package = commands.add_parser("pack", help="Собрать проверенный ZIP проекта")
    package.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "check":
            return 0 if check() else 1
        if args.command == "summary":
            from scope_cache.core import read_analysis

            m = json.loads((ROOT / "models/flash.metadata.json").read_text())
            d = read_analysis()["model"]
            if d["model_sha256"] != m["artifact"]["sha256"]:
                raise ValueError("Диагностика устарела: запустите diagnostics")
            print(
                json.dumps(
                    {
                        "model": m["selected_model"],
                        "test": m["test"],
                        "confidence": d["uncertainty"]["metrics"],
                        "limitations": m["limitations"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "audit":
            from scope_cache.validation import audit

            r = audit(quiet=True)
            print(
                f"{'PASS' if r['ok'] else 'FAIL'}: {r['counts']['checks']} проверок, {r['counts']['events']:,} бросков"
            )
            return 0 if r["ok"] else 1
        if args.command == "score":
            from scope_cache.scoring import read_rows, score as score_one
            from scope_cache.model import load
            from scope_cache.geometry import MapGeometry

            if args.output and (
                args.output.resolve() == args.input.resolve()
                or args.output.resolve().is_relative_to(ROOT / "models")
                or args.output.resolve().is_relative_to(ROOT / "data")
            ):
                raise ValueError("Результат не может перезаписать входные данные или модель")
            rows = read_rows(args.input)
            flash = load() if any(r.get("type") == "Flashbangs" for r in rows) else None
            geometry = MapGeometry() if any(r.get("type") == "Smokes" for r in rows) else None
            values = [score_one(row, geometry, flash, args.mode) for row in rows]
            text = json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            else:
                print(text, end="")
            return 0
        if args.command == "run":
            run(args.source, args.retrain, args.rescore, args.refresh_diagnostics)
            return 0
        if args.command == "train":
            from scope_cache.model import train as train_model

            train_model(include_kernel=not args.without_kernel)
            run(refresh_diagnostics=True)
            return 0
        if args.command == "diagnostics":
            from scope_cache.validation import diagnose_model, diagnose_smoke

            diagnose_model()
            diagnose_smoke()
            return 0
        if args.command == "report":
            from scope_cache.report import execute_report

            execute_report(args.output)
            return 0
        if args.command == "map":
            from scope_cache.geometry import fetch, build_sightlines

            fetch() if args.fetch else build_sightlines()
            return 0
        if args.command == "pack":
            print(json.dumps(pack(args.output), indent=2))
            return 0
        if args.command == "demo":
            from scope_cache.demo import convert, import_demo

            options = {
                "match_id": args.match_id,
                "expected_map": args.expected_map,
                "round_offset": args.round_offset,
                "id_scheme": args.id_scheme,
                "score_smokes": args.score_smokes,
                "allow_geometry_mismatch": args.allow_geometry_mismatch,
                "event_path": ROOT / "data/events.parquet" if args.join_events else None,
            }
            if args.input.suffix.lower() == ".dem":
                with tempfile.TemporaryDirectory(prefix="scope-demo-") as tmp:
                    normalized = Path(tmp) / "source_demo.json.gz"
                    converted = convert(
                        args.input,
                        normalized,
                        expected_map=args.expected_map,
                        sample_ticks=args.sample_ticks,
                    )
                    result = import_demo(normalized, args.output, **options)
                    shutil.copyfile(normalized, args.output / "source_demo.json.gz")
                    (args.output / "demo_conversion.json").write_text(
                        json.dumps(converted, indent=2) + "\n"
                    )
            else:
                result = import_demo(args.input, args.output, **options)
            print(json.dumps(result, indent=2))
            return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"{args.command}: {error}\n")
    return 0

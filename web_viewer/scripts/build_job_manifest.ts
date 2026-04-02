#!/usr/bin/env tsx
import crypto from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import { parse } from "csv-parse/sync";

import type { ZodType } from "zod";

import {
  JobCase,
  JobManifest,
  jobManifestSchema,
  manifestVersion,
  stage1CaseSchema,
  stage1ReportSchema,
  stage2CaseSchema,
  stage2ReportSchema,
  Stage1Case,
  Stage1Report,
  Stage2Case,
  Stage2Report,
} from "../src/lib/manifest/schema";

interface CliOptions {
  jobId: string;
  stage1Dir?: string;
  stage2Dir?: string;
  artifactPrefix: string;
  outputPath?: string;
  datasetOverride?: string;
  launcherUser: string;
  launcherNotes?: string;
}

interface StageMapping {
  localRoot: string;
  remotePrefix: string;
  label: string;
}

interface LoadedStage<TReport extends Stage1Report | Stage2Report, TCase extends Stage1Case | Stage2Case> {
  report: TReport;
  reportSha256: string;
  cases: TCase[];
  mapping: StageMapping;
  outputs: Record<string, unknown> | undefined;
  casesCsvPath: string;
  summaryJsonPath: string;
  reportPath: string;
}

type CsvRecord = Record<string, string>;

type StageLoader<TReport extends Stage1Report | Stage2Report, TCase extends Stage1Case | Stage2Case> = (
  stageDir: string,
  mapping: StageMapping,
) => Promise<LoadedStage<TReport, TCase>>;

const HELP_TEXT = `Usage: pnpm build:manifest -- --job-id <id> [options]

Options:
  --stage1-dir <path>         Path to the stage 1 output directory (contains benchmark_stage_report.json)
  --stage2-dir <path>         Path to the stage 2 output directory (contains benchmark_stage_report.json)
  --artifact-prefix <prefix>  Remote key prefix (default: jobs/<job-id>)
  --output <path>             Write manifest to file instead of stdout
  --launcher-user <name>      User or system launching the job (default: detected from $USER or "unknown")
  --launcher-notes <text>     Optional free-form notes stored in launcher metadata
  --dataset <name>            Override dataset name instead of inferring from stage reports
  --help                      Show this message
`;

async function main(): Promise<void> {
  const cli = parseCli();
  if (!cli.stage1Dir && !cli.stage2Dir) {
    throw new Error("Provide at least --stage1-dir or --stage2-dir");
  }

  const stage1Data = cli.stage1Dir
    ? await loadStage<Stage1Report, Stage1Case>(
        {
          label: "stage1",
          localRoot: cli.stage1Dir,
          remotePrefix: path.posix.join(cli.artifactPrefix, "stage1"),
        },
        stage1ReportSchema,
        stage1CaseSchema,
      )
    : undefined;

  const stage2Data = cli.stage2Dir
    ? await loadStage<Stage2Report, Stage2Case>(
        {
          label: "stage2",
          localRoot: cli.stage2Dir,
          remotePrefix: path.posix.join(cli.artifactPrefix, "stage2"),
        },
        stage2ReportSchema,
        stage2CaseSchema,
      )
    : undefined;

  const dataset =
    cli.datasetOverride ?? stage1Data?.report.dataset ?? stage2Data?.report.dataset ?? (() => {
      throw new Error("Unable to infer dataset. Pass --dataset explicitly.");
    })();

  const createdAt = stage1Data?.report.started_at ?? stage2Data?.report.started_at ?? new Date().toISOString();
  const launcherGit = stage1Data?.report.git ?? stage2Data?.report.git ?? null;
  const launcherCmd = stage1Data?.report.command ?? stage2Data?.report.command ?? [];

  const stage1Cases = stage1Data?.cases ?? [];
  const stage2Cases = stage2Data?.cases ?? [];

  const caseMap = new Map<string, JobCase>();
  for (const entry of stage1Cases) {
    caseMap.set(entry.case_name, { case_name: entry.case_name, stage1: entry });
  }
  for (const entry of stage2Cases) {
    const existing = caseMap.get(entry.case_name);
    if (existing) {
      existing.stage2 = entry;
    } else {
      caseMap.set(entry.case_name, { case_name: entry.case_name, stage2: entry });
    }
  }
  const cases = Array.from(caseMap.values()).sort((a, b) => a.case_name.localeCompare(b.case_name));

  const artifacts = {
    pointcloud_samples: dedupe(stage1Cases.map((c) => c.sample_points_path)),
    mean_implants: dedupe(stage2Cases.map((c) => c.mean_implant_path)),
    eval_metrics: dedupe(stage2Cases.map((c) => c.eval_metrics_path).filter(Boolean) as string[]),
  };

  const manifest: JobManifest = jobManifestSchema.parse({
    schema_version: manifestVersion,
    job_id: cli.jobId,
    dataset,
    created_at: createdAt,
    artifact_prefix: cli.artifactPrefix,
    launcher: {
      user: cli.launcherUser,
      notes: cli.launcherNotes,
      cmd: launcherCmd,
      git: launcherGit,
    },
    stage1: stage1Data
      ? {
          stage_name: stage1Data.report.stage_name,
          stage_dir: stage1Data.mapping.remotePrefix,
          report_path: stage1Data.reportPath,
          report_sha256: stage1Data.reportSha256,
          started_at: stage1Data.report.started_at,
          finished_at: stage1Data.report.finished_at,
          summary: stage1Data.report.summary,
          git: stage1Data.report.git,
          system: stage1Data.report.system,
          outputs: stage1Data.outputs,
        }
      : undefined,
    stage2: stage2Data
      ? {
          stage_name: stage2Data.report.stage_name,
          stage_dir: stage2Data.mapping.remotePrefix,
          report_path: stage2Data.reportPath,
          report_sha256: stage2Data.reportSha256,
          started_at: stage2Data.report.started_at,
          finished_at: stage2Data.report.finished_at,
          summary: stage2Data.report.summary,
          git: stage2Data.report.git,
          system: stage2Data.report.system,
          outputs: stage2Data.outputs,
        }
      : undefined,
    cases,
    artifacts,
  });

  const outputPayload = JSON.stringify(manifest, null, 2) + "\n";
  if (cli.outputPath) {
    await fs.mkdir(path.dirname(cli.outputPath), { recursive: true });
    await fs.writeFile(cli.outputPath, outputPayload, "utf-8");
    console.log(`Wrote manifest to ${cli.outputPath}`);
  } else {
    process.stdout.write(outputPayload);
  }
}

async function loadStage<TReport extends Stage1Report | Stage2Report, TCase extends Stage1Case | Stage2Case>(
  mapping: StageMapping,
  reportSchema: ZodType<TReport>,
  caseSchema: ZodType<TCase>,
): Promise<LoadedStage<TReport, TCase>> {
  const stageDir = mapping.localRoot;
  const reportPath = path.join(stageDir, "benchmark_stage_report.json");
  const summaryPath = path.join(stageDir, "benchmark_summary.json");
  const casesCsvPath = path.join(stageDir, "benchmark_cases.csv");

  const report = await readJson<TReport>(reportPath, reportSchema);
  const reportSha256 = await computeSha256(reportPath);
  const cases = await readCases<TCase>(casesCsvPath, caseSchema, mapping);

  const outputs = normalizeOutputs(report.outputs, mapping);
  const casesCsvRemote = rewriteStagePath(casesCsvPath, mapping);
  const summaryRemote = rewriteStagePath(summaryPath, mapping);
  const reportRemote = rewriteStagePath(reportPath, mapping);
  if (!casesCsvRemote || !summaryRemote || !reportRemote) {
    throw new Error(`Unable to normalize required outputs for ${mapping.label}`);
  }
  outputs.cases_csv = casesCsvRemote;
  outputs.summary_json = summaryRemote;

  return {
    report,
    reportSha256,
    cases,
    mapping,
    outputs,
    summaryJsonPath: summaryRemote,
    casesCsvPath: casesCsvRemote,
    reportPath: reportRemote,
  } as LoadedStage<TReport, TCase>;
}

async function readJson<T>(filePath: string, schema: ZodType<T>): Promise<T> {
  const payload = await fs.readFile(filePath, "utf-8");
  const parsed = JSON.parse(payload);
  return schema.parse(parsed) as T;
}

async function computeSha256(filePath: string): Promise<string> {
  const buf = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(buf).digest("hex");
}

async function readCases<TCase extends Stage1Case | Stage2Case>(
  csvPath: string,
  schema: ZodType<TCase>,
  mapping: StageMapping,
): Promise<TCase[]> {
  const csvText = await fs.readFile(csvPath, "utf-8");
  const records = parse(csvText, { columns: true, skip_empty_lines: true, trim: true }) as CsvRecord[];
  return records.map((record) => {
    const normalized = normalizeCaseRecord(record, mapping);
    return schema.parse(normalized) as TCase;
  });
}

function normalizeCaseRecord(record: CsvRecord, mapping: StageMapping): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    const trimmed = value?.trim?.() ?? "";
    if (
      key.endsWith("_path") ||
      key === "case_dir" ||
      key === "input_points_path" ||
      key === "sample_points_path" ||
      key === "shift_path" ||
      key === "scale_path"
    ) {
      result[key] = rewriteStagePath(trimmed, mapping);
      continue;
    }
    if (isNumericField(key)) {
      result[key] = coerceNumber(trimmed, key);
      continue;
    }
    result[key] = trimmed;
  }
  return result;
}

function isNumericField(key: string): boolean {
  return [
    "num_generated_implants",
    "runtime_sec",
    "gpu_peak_memory_mb",
    "dice",
    "bdice_10mm",
    "hd95_mm",
  ].includes(key);
}

function coerceNumber(value: string, field: string): number | null {
  if (value === "" || value == null) {
    return null;
  }
  const num = Number(value);
  if (Number.isNaN(num)) {
    throw new Error(`Field ${field} expected a number but received ${value}`);
  }
  return num;
}

function rewriteStagePath(input: string | null | undefined, mapping: StageMapping): string | null {
  if (input == null) {
    return null;
  }
  const trimmed = input.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = toPosix(trimmed);
  if (isRemoteUri(normalized) || normalized.startsWith(mapping.remotePrefix)) {
    return normalized;
  }
  const localRootPosix = ensureTrailingSlash(toPosix(mapping.localRoot));
  if (normalized.startsWith(localRootPosix)) {
    const relative = normalized.slice(localRootPosix.length);
    return joinPosix(mapping.remotePrefix, relative);
  }
  const resolved = toPosix(path.resolve(trimmed));
  if (resolved.startsWith(localRootPosix)) {
    const rel = toPosix(path.relative(mapping.localRoot, path.resolve(trimmed)));
    return joinPosix(mapping.remotePrefix, rel);
  }
  if (normalized.startsWith("./") || normalized.startsWith("../")) {
    const rel = normalized.replace(/^\.\//, "");
    return joinPosix(mapping.remotePrefix, rel);
  }
  return normalized;
}

function normalizeOutputs(outputs: Record<string, unknown> | undefined, mapping: StageMapping): Record<string, any> {
  if (!outputs) {
    return {};
  }
  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(outputs)) {
    if (typeof value === "string") {
      normalized[key] = rewriteStagePath(value, mapping);
    } else {
      normalized[key] = value;
    }
  }
  return normalized;
}

function dedupe(values: string[]): string[] {
  const seen = new Set<string>();
  for (const value of values) {
    if (!value) {
      continue;
    }
    seen.add(value);
  }
  return Array.from(seen);
}

function toPosix(value: string): string {
  return value.replace(/\\/g, "/");
}

function joinPosix(...segments: string[]): string {
  return path.posix.join(...segments.filter(Boolean));
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function isRemoteUri(value: string): boolean {
  return value.startsWith("s3://") || value.startsWith("https://") || value.startsWith("http://");
}

function parseCli(): CliOptions {
  const rawArgs = process.argv.slice(2);
  if (rawArgs.includes("--help")) {
    console.log(HELP_TEXT);
    process.exit(0);
  }
  const args: Record<string, string> = {};
  for (let i = 0; i < rawArgs.length; i += 1) {
    const token = rawArgs[i];
    if (!token.startsWith("--")) {
      throw new Error(`Unrecognized argument: ${token}`);
    }
    const [flag, inline] = token.slice(2).split("=", 2);
    if (inline !== undefined) {
      args[flag] = inline;
      continue;
    }
    const next = rawArgs[i + 1];
    if (!next || next.startsWith("--")) {
      args[flag] = "true";
      continue;
    }
    args[flag] = next;
    i += 1;
  }
  const jobId = args["job-id"];
  if (!jobId) {
    throw new Error("--job-id is required");
  }
  const stage1Dir = args["stage1-dir"] ? path.resolve(args["stage1-dir"]) : undefined;
  const stage2Dir = args["stage2-dir"] ? path.resolve(args["stage2-dir"]) : undefined;
  const artifactPrefix = args["artifact-prefix"] ? trimSlashes(args["artifact-prefix"]) : `jobs/${jobId}`;
  const outputPath = args.output ? path.resolve(args.output) : undefined;
  const datasetOverride = args.dataset;
  const launcherUser = args["launcher-user"] ?? process.env.USER ?? "unknown";
  const launcherNotes = args["launcher-notes"];
  return {
    jobId,
    stage1Dir,
    stage2Dir,
    artifactPrefix,
    outputPath,
    datasetOverride,
    launcherUser,
    launcherNotes,
  };
}

function trimSlashes(value: string): string {
  return value.replace(/^\/+/, "").replace(/\/+$/, "");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});

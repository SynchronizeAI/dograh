"use client";

import Link from "next/link";
import { useState } from "react";

import { updateNamedModelConfiguration } from "@/client/sdk.gen";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { detailFromError } from "@/lib/apiError";

import { cleanFallbackPolicy, compatibleConnections, selectionForConnection } from "./configuration";
import { ConfigurationSelect } from "./ConfigurationSelect";
import { AddProviderDialog } from "./ProviderSelect";
import type { ModelConnectionCatalog, NamedModelConfiguration, ProviderConnection, SttFallbackRule } from "./types";

const ASSEMBLYAI_MODEL = "assemblyai-stt-3-7-realtime-agent";

function backupRule(connectionUuid = "", settings: Record<string, unknown> = {}): SttFallbackRule {
    return {
        condition: { type: "error" },
        target: {
            provider_connection_uuid: connectionUuid,
            settings: { ...settings, model: ASSEMBLYAI_MODEL, language: "ko" },
        },
    };
}

export function STTFallbackEditor({ saved, catalog, connections, onConnectionAdded, onSaved }: {
    saved: NamedModelConfiguration;
    catalog: ModelConnectionCatalog;
    connections: ProviderConnection[];
    onConnectionAdded: (connection: ProviderConnection) => void;
    onSaved: () => Promise<void>;
}) {
    const [rules, setRules] = useState<SttFallbackRule[]>(saved.configuration.stt_fallback?.rules || []);
    const [saving, setSaving] = useState(false);
    const [adding, setAdding] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const available = compatibleConnections(catalog, connections, "stt")
        .filter(connection => connection.provider === "assemblyai");
    const rule = rules[0];
    const enabled = Boolean(rule);
    const primary = connections.find(connection => connection.uuid === saved.configuration.stt?.provider_connection_uuid);
    const toggle = (next: boolean) => {
        if (!next) {
            setRules([]);
            return;
        }
        const connection = available[0];
        setRules([backupRule(connection?.uuid)]);
    };
    const selectConnection = (uuid: string) => {
        const connection = available.find(item => item.uuid === uuid);
        if (!connection) return;
        const selection = selectionForConnection(catalog, "stt", connection, rule?.target, connections);
        setRules([backupRule(uuid, selection.settings)]);
    };

    return <form className="space-y-6" onSubmit={async event => {
        event.preventDefault();
        setSaving(true);
        setError(null);
        try {
            const configuration = {
                ...saved.configuration,
                stt_fallback: rules.length
                    ? cleanFallbackPolicy({ version: 1 as const, rules }, catalog, connections, "stt")
                    : null,
            };
            const result = await updateNamedModelConfiguration({
                path: { configuration_uuid: saved.uuid },
                body: { configuration, revision: saved.revision },
            });
            if (result.error) throw new Error(detailFromError(result.error, "Failed to save STT fallback"));
            if (!result.data) throw new Error("Failed to save STT fallback");
            await onSaved();
        } catch (cause) {
            setError(cause instanceof Error ? cause.message : "Failed to save STT fallback");
        } finally {
            setSaving(false);
        }
    }}>
        <div className="space-y-2">
            <p className="text-sm text-muted-foreground">{saved.name}</p>
            <h1 className="text-2xl font-bold">STT fallback</h1>
            <p className="max-w-2xl text-sm text-muted-foreground">Use AssemblyAI Universal-3.7 for Korean recognition if the primary transcriber has an unrecoverable error.</p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-muted/30 p-4">
            <div><p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Primary STT</p><p className="mt-1 font-medium">{primary?.name || "Unavailable connection"}</p></div>
            <Button asChild variant="ghost" size="sm"><Link href={`/model-configurations/${saved.uuid}`}>Edit primary</Link></Button>
        </div>
        {error && <p role="alert" className="rounded-md border border-destructive/40 p-3 text-sm text-destructive">{error}</p>}
        <section className="rounded-lg border bg-card">
            <div className="flex items-start justify-between gap-4 p-5">
                <div className="space-y-1">
                    <Label htmlFor="stt-fallback-enabled" className="text-base font-semibold">AssemblyAI 3.7 backup</Label>
                    <p className="max-w-2xl text-sm text-muted-foreground">The backup uses its own end-of-turn detection and Korean language setting.</p>
                </div>
                <Switch id="stt-fallback-enabled" checked={enabled} disabled={saving} onCheckedChange={toggle} />
            </div>
            {enabled && rule && <div className="space-y-4 border-t p-5">
                <div className="space-y-1.5">
                    <Label htmlFor="stt-fallback-connection">AssemblyAI provider connection</Label>
                    <ConfigurationSelect id="stt-fallback-connection" required value={available.some(item => item.uuid === rule.target.provider_connection_uuid) ? rule.target.provider_connection_uuid : ""}
                        placeholder="Select an AssemblyAI connection"
                        options={available.map(item => ({ value: item.uuid, label: item.name }))}
                        onValueChange={selectConnection} />
                    <Button type="button" variant="outline" size="sm" onClick={() => setAdding(true)}>Add AssemblyAI connection</Button>
                </div>
                <div className="grid gap-4 text-sm sm:grid-cols-2">
                    <div><p className="text-muted-foreground">Model</p><p className="font-medium">{ASSEMBLYAI_MODEL}</p></div>
                    <div><p className="text-muted-foreground">Language</p><p className="font-medium">Korean (ko)</p></div>
                </div>
            </div>}
        </section>
        <div className="flex justify-end border-t pt-5">
            <Button type="submit" disabled={saving || (enabled && !available.some(item => item.uuid === rule?.target.provider_connection_uuid))}>{saving ? "Saving…" : "Save STT fallback"}</Button>
        </div>
        {adding && <AddProviderDialog open onOpenChange={open => setAdding(open)} title="Add AssemblyAI connection"
            catalog={catalog} providerKeys={["assemblyai"]} role="stt" onSaved={connection => {
                onConnectionAdded(connection);
                setAdding(false);
                setRules([backupRule(connection.uuid)]);
            }} />}
    </form>;
}

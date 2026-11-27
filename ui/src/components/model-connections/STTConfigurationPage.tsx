"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { toast } from "sonner";

import { PageShell } from "@/components/layout/PageShell";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useOrgConfig } from "@/context/OrgConfigContext";

import { STTFallbackEditor } from "./STTFallbackEditor";
import { useModelConnections } from "./useModelConnections";

export function STTConfigurationPage({ configurationUuid }: { configurationUuid: string }) {
    const { catalog, connections, configurations, loading, error, reload, addConnection } = useModelConnections();
    const { refreshConfig } = useOrgConfig();
    const saved = configurations.find(item => item.uuid === configurationUuid && item.is_active);
    const primary = connections.find(item => item.uuid === saved?.configuration.stt?.provider_connection_uuid);
    return <PageShell className="space-y-6">
        <Button asChild variant="ghost" className="-ml-3"><Link href={`/model-configurations/${configurationUuid}`}><ArrowLeft className="h-4 w-4" />Back to model configuration</Link></Button>
        {loading && !catalog ? <Skeleton className="h-80 w-full" /> : error ? <div role="alert" className="space-y-3 text-sm text-destructive"><p>{error}</p><Button variant="outline" onClick={() => void reload()}>Retry</Button></div>
            : !saved ? <p role="alert">This model configuration is unavailable.</p>
                : !saved.configuration.stt || primary?.provider === "dograh" ? <p role="alert">STT fallback requires a BYOK STT primary.</p>
                    : catalog && <STTFallbackEditor key={`${saved.uuid}-${saved.revision}`} saved={saved} catalog={catalog} connections={connections} onConnectionAdded={addConnection} onSaved={async () => {
                        await reload();
                        await refreshConfig();
                        toast.success("STT fallback saved");
                    }} />}
    </PageShell>;
}

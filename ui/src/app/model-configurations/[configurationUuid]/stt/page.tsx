import { STTConfigurationPage } from "@/components/model-connections/STTConfigurationPage";

export default async function EditSTTConfigurationPage({ params }: { params: Promise<{ configurationUuid: string }> }) {
    const { configurationUuid } = await params;
    return <STTConfigurationPage configurationUuid={configurationUuid} />;
}

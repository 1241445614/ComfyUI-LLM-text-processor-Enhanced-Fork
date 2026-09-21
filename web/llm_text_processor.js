import { app } from "../../scripts/app.js";

const NODE_CLASSES = ["LLMTextProcessor", "LLMTextProcessorLite"];
const MAX_IMAGE_INPUTS = 10;
const IMAGE_INPUT_NAMES = Array.from(
    { length: MAX_IMAGE_INPUTS },
    (_, i) => `image_${i + 1}`
);

/**
 * 链式图片输入：
 *   - 默认只显示 image_1
 *   - 连上 image_N 后自动追加 image_{N+1}
 *   - 断开尾部插槽时自动移除多余的尾部插槽
 *   - 只在尾部增删，从不移除中间插槽，避免索引错位
 */
function updateImageInputs(node) {
    let maxConnectedIndex = -1;
    for (let i = 0; i < MAX_IMAGE_INPUTS; i++) {
        const input = node.inputs.find(
            inp => inp.name === IMAGE_INPUT_NAMES[i]
        );
        if (input && input.link != null) {
            maxConnectedIndex = Math.max(maxConnectedIndex, i);
        }
    }

    let neededCount = Math.max(maxConnectedIndex + 2, 1);
    neededCount = Math.min(neededCount, MAX_IMAGE_INPUTS);

    const currentImageInputs = node.inputs.filter(
        inp => IMAGE_INPUT_NAMES.includes(inp.name)
    );
    const currentCount = currentImageInputs.length;

    if (currentCount < neededCount) {
        for (let i = currentCount; i < neededCount; i++) {
            node.addInput(IMAGE_INPUT_NAMES[i], "IMAGE");
        }
    } else if (currentCount > neededCount) {
        for (let i = currentCount - 1; i >= neededCount; i--) {
            const name = IMAGE_INPUT_NAMES[i];
            const idx = node.inputs.findIndex(inp => inp.name === name);
            if (idx >= 0) {
                node.removeInput(idx);
            }
        }
    }

    const newSize = node.computeSize();
    node.setSize([node.size[0], newSize[1]]);
    app.graph.setDirtyCanvas(true, true);
}

const PRESET_VALUES = {
    balanced: {
        memory_mode: "auto",
        n_gpu_layers: 99,
        n_cpu_moe_layers: 1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        no_mmap: false,
        cache_ram: 0,
    },
    low_vram_high_ram: {
        memory_mode: "cpu_moe_layers",
        n_gpu_layers: 0,
        n_cpu_moe_layers: 999,
        cache_type_k: "q8_0",
        cache_type_v: "q8_0",
        no_mmap: true,
        cache_ram: 4096,
    },
    dense_27b_low_vram: {
        memory_mode: "gpu_layers",
        n_gpu_layers: 999,
        n_cpu_moe_layers: 1,
        cache_type_k: "q8_0",
        cache_type_v: "q8_0",
        no_mmap: true,
        cache_ram: 8192,
        ffn_offload: "ffn_0_45",
    },
    hybrid_offload: {
        memory_mode: "gpu_and_cpu_moe_layers",
        n_gpu_layers: 20,
        n_cpu_moe_layers: 999,
        cache_type_k: "q8_0",
        cache_type_v: "q8_0",
        no_mmap: true,
        cache_ram: 4096,
    },
    extreme_low_vram: {
        memory_mode: "cpu_moe_layers",
        n_gpu_layers: 0,
        n_cpu_moe_layers: 999,
        cache_type_k: "q4_0",
        cache_type_v: "q4_0",
        no_mmap: true,
        cache_ram: 2048,
    },
};

app.registerExtension({
    name: "ComfyUI.LLMTextProcessor.UI",
    async nodeCreated(node) {
        if (!NODE_CLASSES.includes(node.comfyClass)) return;

        // ---------- 1. 初始只保留 image_1 ----------
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const name = node.inputs[i].name;
            if (
                IMAGE_INPUT_NAMES.includes(name) &&
                name !== IMAGE_INPUT_NAMES[0]
            ) {
                node.removeInput(i);
            }
        }
        if (!node.inputs.find(inp => inp.name === IMAGE_INPUT_NAMES[0])) {
            node.addInput(IMAGE_INPUT_NAMES[0], "IMAGE");
        }

        const initialSize = node.computeSize();
        node.setSize([node.size[0], initialSize[1]]);

        // ---------- 2. 链式更新 ----------
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (
            type, index, connected, link_info, io_slot
        ) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
            const isInput =
                type === 1 ||
                (io_slot && node.inputs && node.inputs.includes(io_slot));
            if (isInput) {
                updateImageInputs(this);
            }
        };

        // ---------- 3. performance_preset 联动 ----------
        const presetWidget = node.widgets?.find(
            w => w.name === "performance_preset"
        );
        if (!presetWidget) return;

        const widgetNames = [
            "memory_mode",
            "n_gpu_layers",
            "n_cpu_moe_layers",
            "cache_type_k",
            "cache_type_v",
            "no_mmap",
            "cache_ram",
            "ffn_offload",
        ];
        const widgets = {};
        for (const name of widgetNames) {
            widgets[name] = node.widgets.find(w => w.name === name);
        }

        const originalPresetCallback = presetWidget.callback;
        presetWidget.callback = function (value) {
            if (originalPresetCallback) originalPresetCallback.call(this, value);

            const preset = PRESET_VALUES[value];
            if (preset) {
                for (const name of widgetNames) {
                    const w = widgets[name];
                    if (w && preset[name] !== undefined) {
                        w.value = preset[name];
                    }
                }
            }

            app.graph.setDirtyCanvas(true, true);
            node.setDirtyCanvas(true, true);
        };
    },
});

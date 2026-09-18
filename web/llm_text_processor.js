import { app } from "../../scripts/app.js";

const NODE_CLASS = "LLMTextProcessor";
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
    // 1. 找出当前已连接的图片插槽中的最大索引
    let maxConnectedIndex = -1;
    for (let i = 0; i < MAX_IMAGE_INPUTS; i++) {
        const input = node.inputs.find(
            inp => inp.name === IMAGE_INPUT_NAMES[i]
        );
        if (input && input.link != null) {
            maxConnectedIndex = Math.max(maxConnectedIndex, i);
        }
    }

    // 2. 需要显示的插槽数量：
    //    最大已连接索引 + 2（保证后面总有一个空位可以连），最少 1 个
    let neededCount = Math.max(maxConnectedIndex + 2, 1);
    neededCount = Math.min(neededCount, MAX_IMAGE_INPUTS);

    // 3. 当前图片插槽数量
    const currentImageInputs = node.inputs.filter(
        inp => IMAGE_INPUT_NAMES.includes(inp.name)
    );
    const currentCount = currentImageInputs.length;

    // 4. 只在尾部增删
    if (currentCount < neededCount) {
        for (let i = currentCount; i < neededCount; i++) {
            node.addInput(IMAGE_INPUT_NAMES[i], "IMAGE");
        }
    } else if (currentCount > neededCount) {
        // 从后往前移除尾部插槽（它们应该是空的）
        for (let i = currentCount - 1; i >= neededCount; i--) {
            const name = IMAGE_INPUT_NAMES[i];
            const idx = node.inputs.findIndex(inp => inp.name === name);
            if (idx >= 0) {
                node.removeInput(idx);
            }
        }
    }

    // 5. 调整尺寸并重绘
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
        if (node.comfyClass !== NODE_CLASS) return;

        // ---------- 1. 初始只保留 image_1 ----------
        // 后端声明了 image_1..image_10，但前端节点初始会被填充所有输入，
        // 所以先把 image_2..image_10 全部移除，只留 image_1。
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const name = node.inputs[i].name;
            if (
                IMAGE_INPUT_NAMES.includes(name) &&
                name !== IMAGE_INPUT_NAMES[0]
            ) {
                node.removeInput(i);
            }
        }
        // 确保 image_1 存在
        if (!node.inputs.find(inp => inp.name === IMAGE_INPUT_NAMES[0])) {
            node.addInput(IMAGE_INPUT_NAMES[0], "IMAGE");
        }

        // 调整尺寸
        const initialSize = node.computeSize();
        node.setSize([node.size[0], initialSize[1]]);

        // ---------- 2. 监听连接变化，链式更新 ----------
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (
            type, index, connected, link_info, io_slot
        ) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
            // 只处理输入侧
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
import { app } from "../../../scripts/app.js";

/* =====================================================================
 * Sokes 🦬 frontend extensions
 * Works in both the legacy LiteGraph canvas and Nodes 2.0 (Vue nodes).
 *
 * Rules followed for Nodes 2.0 compatibility:
 *  - Buttons are plain `addWidget("button", ...)` widgets (no custom type,
 *    no DOM swapping). Change the text with `widget.label`.
 *  - No reliance on `node.element` (doesn't exist for Vue nodes).
 *  - Hidden-noodle state is stored in node.properties so it survives reloads.
 * ===================================================================== */

const GLOBAL_TYPE = "Global Folder and Project Settings | sokes 🦬";
const SAVE_TYPE = "Save File Path and Name | sokes 🦬";
const LINKED_SLOTS = ["main_folder", "project_name", "date_format"];

/* ---------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------- */

function addButton(node, name, label, onClick) {
    const w = node.addWidget("button", name, null, () => onClick(w), { serialize: false });
    w.label = label;
    w.serialize = false;
    return w;
}

function flashLabel(node, widget, text, resetText, ms = 2000) {
    widget.label = text;
    node.setDirtyCanvas?.(true, true);
    setTimeout(() => {
        widget.label = resetText;
        node.setDirtyCanvas?.(true, true);
    }, ms);
}

/* ---------------------------------------------------------------------
 * Hidden noodles
 * A link is "hidden" if its target input name is listed in
 * target.properties.sokes_hidden_inputs. Hidden links are drawn only while
 * either end node is hovered or selected.
 * ------------------------------------------------------------------- */

const hoveredNodeIds = new Set();

function isHiddenLink(link) {
    if (!link) return false;
    const graph = app.graph;
    const target = graph?.getNodeById?.(link.target_id);
    const inputName = target?.inputs?.[link.target_slot]?.name;
    const list = target?.properties?.sokes_hidden_inputs;
    return Array.isArray(list) && inputName !== undefined && list.includes(inputName);
}

function shouldRevealLink(link) {
    if (hoveredNodeIds.has(link.origin_id) || hoveredNodeIds.has(link.target_id)) return true;
    const sel = app.canvas?.selected_nodes;
    return !!(sel && (sel[link.origin_id] || sel[link.target_id]));
}

function setupHoverTracking(node) {
    if (node._sokesHoverSetup) return;
    node._sokesHoverSetup = true;

    const prevEnter = node.onMouseEnter;
    node.onMouseEnter = function () {
        hoveredNodeIds.add(this.id);
        app.canvas?.setDirty(true, true);
        return prevEnter?.apply(this, arguments);
    };

    const prevLeave = node.onMouseLeave;
    node.onMouseLeave = function () {
        hoveredNodeIds.delete(this.id);
        app.canvas?.setDirty(true, true);
        return prevLeave?.apply(this, arguments);
    };

    // When an input is disconnected, forget that it was "hidden" so a later
    // manual connection draws normally.
    const prevConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected) {
        const r = prevConn?.apply(this, arguments);
        // type 1 === LiteGraph.INPUT
        if (type === 1 && !connected) {
            const name = this.inputs?.[index]?.name;
            const list = this.properties?.sokes_hidden_inputs;
            if (Array.isArray(list) && name) {
                this.properties.sokes_hidden_inputs = list.filter((n) => n !== name);
            }
        }
        return r;
    };

    const prevRemoved = node.onRemoved;
    node.onRemoved = function () {
        hoveredNodeIds.delete(this.id);
        return prevRemoved?.apply(this, arguments);
    };
}

/** Connect src.output[srcSlot] -> dst.input[dstSlot] and mark the link hidden. */
function connectHidden(src, srcSlot, dst, dstSlot) {
    const result = src.connect(srcSlot, dst, dstSlot);
    if (!result) return false;

    const inputName = dst.inputs?.[dstSlot]?.name;
    if (inputName) {
        dst.properties = dst.properties || {};
        const list = (dst.properties.sokes_hidden_inputs = dst.properties.sokes_hidden_inputs || []);
        if (!list.includes(inputName)) list.push(inputName);
    }
    setupHoverTracking(src);
    setupHoverTracking(dst);
    app.canvas?.setDirty(true, true);
    return true;
}

/** Connect all matching slots from a Global node to a Save node. Returns #new links. */
function connectGlobalToSave(globalNode, saveNode) {
    let count = 0;
    for (const name of LINKED_SLOTS) {
        const outIdx = globalNode.outputs?.findIndex((o) => o.name === name) ?? -1;
        const inIdx = saveNode.inputs?.findIndex((i) => i.name === name) ?? -1;
        if (outIdx === -1 || inIdx === -1) continue;

        const input = saveNode.inputs[inIdx];
        if (input.link !== null && input.link !== undefined) continue; // already connected

        if (connectHidden(globalNode, outIdx, saveNode, inIdx)) count++;
    }
    return count;
}

/* ---------------------------------------------------------------------
 * Patch link rendering so hidden links are skipped unless revealed.
 * Modern frontend: LGraphCanvas.prototype.renderLink(ctx, a, b, link, ...)
 * ------------------------------------------------------------------- */

app.registerExtension({
    name: "Sokes.InvisibleNoodles.Render",

    async setup() {
        const proto = LGraphCanvas?.prototype;
        if (!proto || proto._sokesRenderLinkPatched) return;

        const originalRenderLink = proto.renderLink;
        if (typeof originalRenderLink !== "function") {
            console.warn("[Sokes] LGraphCanvas.renderLink not found; hidden noodles disabled.");
            return;
        }
        proto._sokesRenderLinkPatched = true;

        proto.renderLink = function (ctx, a, b, link, ...rest) {
            if (link && isHiddenLink(link) && !shouldRevealLink(link)) {
                return; // skip drawing
            }
            return originalRenderLink.call(this, ctx, a, b, link, ...rest);
        };
    },
});

/* ---------------------------------------------------------------------
 * Street View Loader: heading/pitch drag area (unchanged logic)
 * ------------------------------------------------------------------- */

app.registerExtension({
    name: "Sokes.StreetViewLoader.Dragger",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Street View Loader | sokes 🦬") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            setTimeout(() => {
                const headingWidget = this.widgets.find((w) => w.name === "heading");
                const pitchWidget = this.widgets.find((w) => w.name === "pitch");

                if (!headingWidget || !pitchWidget) {
                    console.error("StreetViewLoader Error: Could not find heading or pitch widgets.");
                    return;
                }

                const dragArea = document.createElement("div");
                Object.assign(dragArea.style, {
                    width: "100%",
                    minHeight: "100px",
                    backgroundColor: "#222",
                    cursor: "grab",
                    border: "1px dashed var(--border-color)",
                    borderRadius: "4px",
                    marginTop: "10px",
                    position: "relative",
                    overflow: "hidden",
                });

                const dot = document.createElement("div");
                Object.assign(dot.style, {
                    width: "10px",
                    height: "10px",
                    backgroundColor: "red",
                    borderRadius: "50%",
                    position: "absolute",
                    transform: "translate(-50%, -50%)",
                    pointerEvents: "none",
                });
                dragArea.appendChild(dot);

                const updateDotPosition = () => {
                    const x = (headingWidget.value / 360) * 100;
                    const y = ((-pitchWidget.value + 90) / 180) * 100;
                    dot.style.left = `${x}%`;
                    dot.style.top = `${y}%`;
                };

                if (headingWidget.element?.parentElement) {
                    headingWidget.element.parentElement.style.display = "none";
                }
                if (pitchWidget.element?.parentElement) {
                    pitchWidget.element.parentElement.style.display = "none";
                }

                this.addDOMWidget("heading_pitch_dragger", "preview", dragArea);

                dragArea.onmousedown = (e) => {
                    e.preventDefault();
                    let lastX = e.clientX;
                    let lastY = e.clientY;
                    dragArea.style.cursor = "grabbing";

                    const onMouseMove = (moveE) => {
                        const deltaX = moveE.clientX - lastX;
                        const deltaY = moveE.clientY - lastY;
                        lastX = moveE.clientX;
                        lastY = moveE.clientY;

                        let newHeading = (headingWidget.value + deltaX * 0.5) % 360;
                        if (newHeading < 0) newHeading += 360;
                        headingWidget.value = parseFloat(newHeading.toFixed(1));

                        const newPitch = pitchWidget.value - deltaY * 0.5;
                        pitchWidget.value = parseFloat(Math.max(-90, Math.min(90, newPitch)).toFixed(1));

                        if (headingWidget.callback) headingWidget.callback(headingWidget.value);
                        if (pitchWidget.callback) pitchWidget.callback(pitchWidget.value);

                        updateDotPosition();
                    };

                    const onMouseUp = () => {
                        dragArea.style.cursor = "grab";
                        document.removeEventListener("mousemove", onMouseMove);
                        document.removeEventListener("mouseup", onMouseUp);
                    };

                    document.addEventListener("mousemove", onMouseMove);
                    document.addEventListener("mouseup", onMouseUp);
                };

                updateDotPosition();
            }, 10);
        };
    },
});

/* ---------------------------------------------------------------------
 * Save File Path and Name: "Connect to Global Folder Settings" button
 * ------------------------------------------------------------------- */

app.registerExtension({
    name: "Sokes.SaveFilePath.AutoConnect",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== SAVE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            setupHoverTracking(this);

            const idleLabel = "🔗 Connect to Global Folder Settings";

            addButton(this, "auto_connect_global", idleLabel, (widget) => {
                const graph = app.graph;

                // Find or create the Global node
                let globalNode = graph._nodes.find((n) => n.type === GLOBAL_TYPE);
                if (!globalNode) {
                    globalNode = LiteGraph.createNode(GLOBAL_TYPE);
                    if (!globalNode) {
                        flashLabel(this, widget, "⚠️ Global node type not found", idleLabel);
                        return;
                    }
                    graph.add(globalNode);
                    globalNode.pos = [this.pos[0] - 20, this.pos[1] - globalNode.size[1] - 20];
                }

                const connected = connectGlobalToSave(globalNode, this);
                flashLabel(
                    this,
                    widget,
                    connected > 0 ? `✅ Connected ${connected} input(s)` : "🔗 Already connected",
                    idleLabel
                );
            });
        };
    },
});

/* ---------------------------------------------------------------------
 * Global Folder and Project Settings: "Auto-Connect Save File Path Nodes"
 * ------------------------------------------------------------------- */

app.registerExtension({
    name: "Sokes.GlobalFolderSettings.AutoConnect",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== GLOBAL_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            setupHoverTracking(this);

            const idleLabel = "🔗 Auto-Connect Save File Path Nodes";

            addButton(this, "auto_connect", idleLabel, (widget) => {
                const graph = app.graph;

                let saveNodes = graph._nodes.filter((n) => n.type === SAVE_TYPE && n !== this);

                // Create one if none exist
                if (saveNodes.length === 0) {
                    const newNode = LiteGraph.createNode(SAVE_TYPE);
                    if (!newNode) {
                        flashLabel(this, widget, "⚠️ Save node type not found", idleLabel);
                        return;
                    }
                    graph.add(newNode);
                    newNode.pos = [this.pos[0] + this.size[0] + 20, this.pos[1]];
                    saveNodes = [newNode];
                }

                let connected = 0;
                for (const node of saveNodes) {
                    connected += connectGlobalToSave(this, node);
                }

                flashLabel(
                    this,
                    widget,
                    connected > 0 ? `✅ Connected ${connected} input(s)` : "🔗 All nodes already connected",
                    idleLabel
                );
            });
        };
    },
});

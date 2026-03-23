import { app } from "../../../scripts/app.js";

// CSS for invisible noodles with glow effects
const SOKES_STYLES = `
<style id="sokes-invisible-noodles-style">
/* Hide the noodle/line for sokes-hidden-link connections */
.sokes-hidden-link .litegraph.link {
    stroke: transparent !important;
    stroke-width: 0 !important;
    pointer-events: none;
    opacity: 0;
}

/* Glow effect on connected slots when hovering parent node */
.sokes-node-hover .sokes-connected-slot-output::before,
.sokes-node-hover .sokes-connected-slot-input::before {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    box-shadow: 0 0 10px 3px var(--sokes-link-color, #4a9eff),
                0 0 20px 6px var(--sokes-link-color, #4a9eff);
    animation: sokes-pulse 1.5s ease-in-out infinite;
}

/* Output slot glow styling */
.sokes-connected-slot-output {
    position: relative;
}
.sokes-connected-slot-output .slot_circle {
    stroke: var(--sokes-link-color, #4a9eff) !important;
    stroke-width: 3px !important;
    filter: drop-shadow(0 0 5px var(--sokes-link-color, #4a9eff));
    transition: all 0.3s ease;
}

/* Input slot glow styling */
.sokes-connected-slot-input {
    position: relative;
}
.sokes-connected-slot-input .slot_label {
    color: var(--sokes-link-color, #4a9eff) !important;
    text-shadow: 0 0 5px var(--sokes-link-color, #4a9eff);
    font-weight: bold;
}
.sokes-connected-slot-input .litegraph.io_slot {
    position: relative;
}
.sokes-connected-slot-input .litegraph.io_slot::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid var(--sokes-link-color, #4a9eff);
    box-shadow: 0 0 8px 2px var(--sokes-link-color, #4a9eff);
}

/* Pulse animation for glow effect */
@keyframes sokes-pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.7;
        transform: scale(1.1);
    }
}

/* When hovering a connected node, show the noodle briefly */
.sokes-link-visible .litegraph.link {
    stroke: var(--sokes-link-color, #4a9eff) !important;
    stroke-width: 3px !important;
    opacity: 0.6 !important;
    pointer-events: auto;
}

/* Highlight slot backgrounds when connected */
.sokes-connected-slot-output.bg {
    background: var(--sokes-link-color, #4a9eff) !important;
}
</style>
`;

// Helper function to create custom link with hidden noodle styling
function createHiddenLink(sokesNode, targetNode, fromSlot, toSlot, linkColor = '#4a9eff') {
    // Create the connection normally first
    const result = sokesNode.connect(fromSlot, targetNode, toSlot);

    if (result && result.link) {
        const linkId = result.link;

        // Mark this as a hidden link by adding custom data
        if (!app.graph.links[linkId]) {
            return result;
        }

        const link = app.graph.links[linkId];
        link._sokesHidden = true;
        link._sokesColor = linkColor;

        // Add class to both nodes for slot styling
        sokesNode.element.classList.add('sokes-node-with-hidden-links');
        targetNode.element.classList.add('sokes-node-with-hidden-links');

        // Style the output slot on source node
        const outputSlot = sokesNode.outputs[fromSlot];
        if (outputSlot) {
            outputSlot._sokesConnected = true;
            outputSlot._sokesColor = linkColor;
        }

        // Style the input slot on target node
        const inputSlot = targetNode.inputs[toSlot];
        if (inputSlot) {
            inputSlot._sokesConnected = true;
            inputSlot._sokesColor = linkColor;
        }
    }

    return result;
}

// Apply hover effects to nodes with hidden links
function setupNodeHoverEffects(node) {
    const element = node.element;
    if (!element || element._sokesHoverSetup) return;
    element._sokesHoverSetup = true;

    // Show glows on mouse enter
    element.addEventListener('mouseenter', () => {
        element.classList.add('sokes-node-hover');

        // Activate sokes hover mode - this makes hidden links visible
        app.canvas.sokesHoverActive = true;

        // Apply glow styling to connected slots
        if (node.outputs) {
            node.outputs.forEach((output, index) => {
                if (output._sokesConnected) {
                    // Find the DOM element for this output slot and add glow class
                    const slotElement = element.querySelector(`[data-output-index="${index}"]`);
                    if (slotElement) {
                        slotElement.classList.add('sokes-connected-slot-output');
                        slotElement.style.setProperty('--sokes-link-color', output._sokesColor || '#4a9eff');
                    }
                }
            });
        }

        if (node.inputs) {
            node.inputs.forEach((input, index) => {
                if (input._sokesConnected && input.link !== null && input.link !== undefined) {
                    const slotElement = element.querySelector(`[data-input-index="${index}"]`);
                    if (slotElement) {
                        slotElement.classList.add('sokes-connected-slot-input');
                        slotElement.style.setProperty('--sokes-link-color', input._sokesColor || '#4a9eff');
                    }
                }
            });
        }

        // Force canvas redraw to show links
        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }
    });

    // Hide glows on mouse leave
    element.addEventListener('mouseleave', () => {
        element.classList.remove('sokes-node-hover');

        // Check if any other sokes node is still hovered
        let stillHovering = false;
        document.querySelectorAll('.sokes-node-with-hidden-links').forEach(el => {
            if (el.classList.contains('sokes-node-hover')) {
                stillHovering = true;
            }
        });

        if (!stillHovering) {
            app.canvas.sokesHoverActive = false;
        }

        // Remove glow styling from slots
        if (node.outputs) {
            node.outputs.forEach((output, index) => {
                const slotElement = element.querySelector(`[data-output-index="${index}"]`);
                if (slotElement) {
                    slotElement.classList.remove('sokes-connected-slot-output');
                }
            });
        }

        if (node.inputs) {
            node.inputs.forEach((input, index) => {
                const slotElement = element.querySelector(`[data-input-index="${index}"]`);
                if (slotElement) {
                    slotElement.classList.remove('sokes-connected-slot-input');
                }
            });
        }

        // Force canvas redraw to hide links again
        if (app.canvas) {
            app.canvas.setDirty(true, true);
        }
    });
}

// Inject styles once
if (!document.getElementById('sokes-invisible-noodles-style')) {
    document.head.insertAdjacentHTML('beforeend', SOKES_STYLES);
}

// Extension to handle invisible noodles - patches litegraph's Link class and canvas rendering
app.registerExtension({
    name: "Sokes.InvisibleNoodles.Render",

    async setup() {
        // Store original drawLink function
        const originalDrawLink = LGraphCanvas.prototype.draw_link_on_canvas;

        // Override to support hidden sokes links with glow color
        if (originalDrawLink && !LGraphCanvas._sokesPatched) {
            LGraphCanvas._sokesPatched = true;

            LGraphCanvas.prototype.draw_link_on_canvas = function(canvas, start_point, end_point, color, fixed_color) {
                // Check if this is a sokes hidden link - skip drawing the line itself
                // The visual feedback will be the glow on slots

                const ctx = canvas.getContext ? canvas : app.canvas.context;

                // For sokes hidden links, draw with opacity 0 (invisible)
                // We still need to call original for proper rendering system
                ctx.save();

                // Set alpha to 0.01 (nearly invisible but still interacts with hover detection if needed)
                ctx.globalAlpha = 0.01;

                originalDrawLink.call(this, canvas, start_point, end_point, color, fixed_color);

                ctx.restore();
            };
        }

        // Also patch the main draw_link method for color handling
        const originalDrawLinkMain = LGraphCanvas.prototype.draw_link;
        if (originalDrawLinkMain && !LGraphCanvas._sokesDrawLinkPatched) {
            LGraphCanvas._sokesDrawLinkPatched = true;

            LGraphCanvas.prototype.draw_link = function(link_info, canvas, skip_nodes, color, time) {
                const link = app.graph.links[link_info.id];

                // For sokes hidden links, pass a flag to make them invisible
                if (link?._sokesHidden && !this.sokesHoverActive) {
                    // Draw with very low opacity - essentially invisible
                    const prevAlpha = this.ctx?.globalAlpha || 1;
                    if (this.ctx) this.ctx.globalAlpha = 0.01;
                    const result = originalDrawLinkMain.call(this, link_info, canvas, skip_nodes, color, time);
                    if (this.ctx) this.ctx.globalAlpha = prevAlpha;
                    return result;
                }

                return originalDrawLinkMain.call(this, link_info, canvas, skip_nodes, color, time);
            };
        }

        // Override getLinkColor to use sokes custom colors
        const originalGetLinkColor = LGraphCanvas.prototype.getLinkColor;
        LGraphCanvas.prototype.getLinkColor = function(link) {
            if (link?._sokesColor) {
                return link._sokesColor;
            }
            return originalGetLinkColor.call(this, link);
        };
    }
});

app.registerExtension({
    name: "Sokes.StreetViewLoader.Dragger",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Street View Loader | sokes 🦬") {
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
                        overflow: "hidden"
                    });

                    const dot = document.createElement("div");
                    Object.assign(dot.style, {
                        width: "10px",
                        height: "10px",
                        backgroundColor: "red",
                        borderRadius: "50%",
                        position: "absolute",
                        transform: "translate(-50%, -50%)",
                        pointerEvents: "none"
                    });
                    dragArea.appendChild(dot);

                    const updateDotPosition = () => {
                        const x = (headingWidget.value / 360) * 100;
                        // Invert the Y-axis mapping to match mouse movement
                        const y = ((-pitchWidget.value + 90) / 180) * 100;
                        dot.style.left = `${x}%`;
                        dot.style.top = `${y}%`;
                    };

                    if (headingWidget.element && headingWidget.element.parentElement) {
                        headingWidget.element.parentElement.style.display = "none";
                    }
                    if (pitchWidget.element && pitchWidget.element.parentElement) {
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

                            let newPitch = pitchWidget.value - deltaY * 0.5;
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
        }
    }
});

app.registerExtension({
    name: "Sokes.SaveFilePath.AutoConnect",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Save File Path and Name | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Add Auto-Connect button
                const connectButton = document.createElement("button");
                connectButton.textContent = "🔗 Connect to Global Folder Settings";
                Object.assign(connectButton.style, {
                    width: "100%",
                    padding: "8px 12px",
                    marginTop: "10px",
                    backgroundColor: "#4a9eff",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontSize: "12px",
                    fontWeight: "bold"
                });

                this.addWidget("button", "auto_connect_global", "Auto-Connect", () => {
                    this.autoConnectToGlobalFolder();
                }).type = "custom";

                // Replace the default widget element with our button
                const btnWidget = this.widgets.find(w => w.name === "auto_connect_global");
                if (btnWidget) {
                    setTimeout(() => {
                        if (btnWidget.element && btnWidget.element.parentElement) {
                            const container = btnWidget.element.parentElement;
                            container.innerHTML = "";
                            container.appendChild(connectButton);

                            connectButton.onclick = () => {
                                this.autoConnectToGlobalFolder();
                            };
                        }
                    }, 50);
                }

                // Method to auto-connect to Global Folder Settings node with hidden noodles
                this.autoConnectToGlobalFolder = function() {
                    const graph = app.graph;
                    let connectedCount = 0;
                    let globalNode = null;

                    console.log("[Sokes Auto-Connect] Save File Path connecting to Global...", {
                        nodeId: this.id,
                        nodeInputs: this.inputs,
                        totalNodes: graph._nodes.length
                    });

                    // Find existing Global Folder Settings node
                    for (let i = 0; i < graph._nodes.length; i++) {
                        const candidate = graph._nodes[i];
                        if (candidate.type === "Global Folder and Project Settings | sokes 🦬") {
                            globalNode = candidate;
                            console.log("[Sokes Auto-Connect] Found existing Global Folder node:", candidate.id);
                            break;
                        }
                    }

                    // If no Global Folder node exists, create one
                    if (!globalNode) {
                        console.log("[Sokes Auto-Connect] No Global Folder Settings node found, creating one...");
                        globalNode = LiteGraph.createNode("Global Folder and Project Settings | sokes 🦬");
                        graph.add(globalNode);

                        // Position the new node above and slightly to the left of this Save File Path node
                        globalNode.pos = [this.pos[0] - 20, this.pos[1] - globalNode.size[1] - 20];

                        // Highlight the new node briefly
                        if (globalNode.flags?.collapsible) globalNode.flags.collapsible = true;
                        app.canvas?.draw(true, true);
                    }

                    // Find output slot INDEX on Global Folder node
                    const mainFolderOutputIndex = globalNode.outputs?.findIndex(out => out.name === "main_folder");
                    const projectOutputIndex = globalNode.outputs?.findIndex(out => out.name === "project_name");
                    const dateOutputIndex = globalNode.outputs?.findIndex(out => out.name === "date_format");

                    // Find input slot INDEX on this Save File Path node
                    const mainFolderInputIndex = this.inputs?.findIndex(inp => inp.name === "main_folder");
                    const projectInputIndex = this.inputs?.findIndex(inp => inp.name === "project_name");
                    const dateInputIndex = this.inputs?.findIndex(inp => inp.name === "date_format");

                    if (mainFolderOutputIndex !== -1 && mainFolderInputIndex !== -1) {
                        // Connect main_folder if not already connected (with hidden noodle)
                        const inputSlot = this.inputs[mainFolderInputIndex];
                        if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                            console.log("[Sokes Auto-Connect] Connecting main_folder...");
                            createHiddenLink(globalNode, this, mainFolderOutputIndex, mainFolderInputIndex);
                            setupNodeHoverEffects(globalNode);
                            setupNodeHoverEffects(this);
                            connectedCount++;
                        }
                    }

                    if (projectOutputIndex !== -1 && projectInputIndex !== -1) {
                        // Connect project_name if not already connected (with hidden noodle)
                        const inputSlot = this.inputs[projectInputIndex];
                        if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                            console.log("[Sokes Auto-Connect] Connecting project_name...");
                            createHiddenLink(globalNode, this, projectOutputIndex, projectInputIndex);
                            setupNodeHoverEffects(globalNode);
                            setupNodeHoverEffects(this);
                            connectedCount++;
                        }
                    }

                    if (dateOutputIndex !== -1 && dateInputIndex !== -1) {
                        // Connect date_format if not already connected (with hidden noodle)
                        const inputSlot = this.inputs[dateInputIndex];
                        if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                            console.log("[Sokes Auto-Connect] Connecting date_format...");
                            createHiddenLink(globalNode, this, dateOutputIndex, dateInputIndex);
                            setupNodeHoverEffects(globalNode);
                            setupNodeHoverEffects(this);
                            connectedCount++;
                        }
                    }

                    // Update button text to show result
                    connectButton.textContent = connectedCount > 0
                        ? `✅ Connected ${connectedCount} input(s)`
                        : "🔗 Already connected or no Global Folder Settings found";

                    setTimeout(() => {
                        connectButton.textContent = "🔗 Connect to Global Folder Settings";
                    }, 2000);
                };
            };
        }
    }
});

app.registerExtension({
    name: "Sokes.GlobalFolderSettings.AutoConnect",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Global Folder and Project Settings | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Add Auto-Connect button
                const connectButton = document.createElement("button");
                connectButton.textContent = "🔗 Auto-Connect Save File Path Nodes";
                Object.assign(connectButton.style, {
                    width: "100%",
                    padding: "8px 12px",
                    marginTop: "10px",
                    backgroundColor: "#4a9eff",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontSize: "12px",
                    fontWeight: "bold"
                });

                connectButton.onmousedown = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.autoConnectSaveFilePathNodes();
                };

                this.addWidget("button", "auto_connect", "Auto-Connect", () => {
                    this.autoConnectSaveFilePathNodes();
                }).type = "custom";

                // Replace the default widget element with our button
                const btnWidget = this.widgets.find(w => w.name === "auto_connect");
                if (btnWidget) {
                    setTimeout(() => {
                        if (btnWidget.element && btnWidget.element.parentElement) {
                            const container = btnWidget.element.parentElement;
                            container.innerHTML = "";
                            container.appendChild(connectButton);

                            connectButton.onclick = () => {
                                this.autoConnectSaveFilePathNodes();
                            };
                        }
                    }, 50);
                }

                // Method to auto-connect all Save File Path nodes with hidden noodles
                this.autoConnectSaveFilePathNodes = function() {
                    const graph = app.graph;
                    let connectedCount = 0;

                    console.log("[Sokes Auto-Connect] Starting...", {
                        thisOutputs: this.outputs,
                        graphNodesCount: graph._nodes.length
                    });

                    // Find all Save File Path and Name nodes
                    let saveFilePathNodes = [];
                    for (let i = 0; i < graph._nodes.length; i++) {
                        const node = graph._nodes[i];
                        if (node.type === "Save File Path and Name | sokes 🦬") {
                            saveFilePathNodes.push(node);
                        }
                    }

                    // If no Save File Path nodes exist, create one
                    if (saveFilePathNodes.length === 0) {
                        console.log("[Sokes Auto-Connect] No Save File Path nodes found, creating one...");
                        const newNode = LiteGraph.createNode("Save File Path and Name | sokes 🦬");
                        graph.add(newNode);

                        // Position the new node to the right of the Global Folder node
                        newNode.pos = [this.pos[0] + this.size[0] + 20, this.pos[1]];

                        // Add the newly created node to our list
                        saveFilePathNodes.push(newNode);

                        // Highlight the new node briefly
                        newNode.flags?.collapsible?.set?.(true);
                        app.canvas?.draw(true, true);
                    }

                    // Now connect all Save File Path nodes
                    for (let i = 0; i < saveFilePathNodes.length; i++) {
                        const node = saveFilePathNodes[i];
                        if (node === this) continue;

                        // Find output slot INDEX on this Global Folder node
                            const mainFolderOutputIndex = this.outputs?.findIndex(out => out.name === "main_folder");
                            const projectOutputIndex = this.outputs?.findIndex(out => out.name === "project_name");
                            const dateOutputIndex = this.outputs?.findIndex(out => out.name === "date_format");

                            // Find input slot INDEX on Save File Path node
                            const mainFolderInputIndex = node.inputs?.findIndex(inp => inp.name === "main_folder");
                            const projectInputIndex = node.inputs?.findIndex(inp => inp.name === "project_name");
                            const dateInputIndex = node.inputs?.findIndex(inp => inp.name === "date_format");

                            console.log("[Sokes Auto-Connect] Found Save File Path node:", {
                                nodeId: node.id,
                                mainFolderOutputIndex,
                                projectOutputIndex,
                                dateOutputIndex,
                                mainFolderInputIndex,
                                projectInputIndex,
                                dateInputIndex,
                                mainFolderHasLink: node.inputs?.[mainFolderInputIndex]?.link,
                                projectHasLink: node.inputs?.[projectInputIndex]?.link,
                                dateHasLink: node.inputs?.[dateInputIndex]?.link
                            });

                            if (mainFolderOutputIndex !== -1 && mainFolderInputIndex !== -1) {
                                // Connect main_folder if not already connected (with hidden noodle)
                                const inputSlot = node.inputs[mainFolderInputIndex];
                                if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                                    console.log("[Sokes Auto-Connect] Connecting main_folder...", {
                                        fromNode: this.id,
                                        fromSlot: mainFolderOutputIndex,
                                        toNode: node.id,
                                        toSlot: mainFolderInputIndex
                                    });
                                    createHiddenLink(this, node, mainFolderOutputIndex, mainFolderInputIndex);
                                    setupNodeHoverEffects(this);
                                    setupNodeHoverEffects(node);
                                    connectedCount++;
                                }
                            }

                            if (projectOutputIndex !== -1 && projectInputIndex !== -1) {
                                // Connect project_name if not already connected (with hidden noodle)
                                const inputSlot = node.inputs[projectInputIndex];
                                if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                                    console.log("[Sokes Auto-Connect] Connecting project_name...", {
                                        fromNode: this.id,
                                        fromSlot: projectOutputIndex,
                                        toNode: node.id,
                                        toSlot: projectInputIndex
                                    });
                                    createHiddenLink(this, node, projectOutputIndex, projectInputIndex);
                                    setupNodeHoverEffects(this);
                                    setupNodeHoverEffects(node);
                                    connectedCount++;
                                }
                            }

                            if (dateOutputIndex !== -1 && dateInputIndex !== -1) {
                                // Connect date_format if not already connected (with hidden noodle)
                                const inputSlot = node.inputs[dateInputIndex];
                                if (!inputSlot || inputSlot.link === null || inputSlot.link === undefined) {
                                    console.log("[Sokes Auto-Connect] Connecting date_format...", {
                                        fromNode: this.id,
                                        fromSlot: dateOutputIndex,
                                        toNode: node.id,
                                        toSlot: dateInputIndex
                                    });
                                    createHiddenLink(this, node, dateOutputIndex, dateInputIndex);
                                    setupNodeHoverEffects(this);
                                    setupNodeHoverEffects(node);
                                    connectedCount++;
                                }
                            }
                    }

                    // Update button text to show result
                    connectButton.textContent = connectedCount > 0
                        ? `✅ Connected ${connectedCount} input(s)`
                        : "🔗 All nodes already connected";

                    setTimeout(() => {
                        connectButton.textContent = "🔗 Auto-Connect All Save File Path Nodes";
                    }, 2000);
                };
            };
        }
    }
});

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "sokes.ColorWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        /**
         * Node: Hex Color Swatch
         * Displays multiple, wrapping color swatches from a comma-separated string.
         */
        if (nodeData.name === "Hex Color Swatch | sokes 🦬") {

            // --- Define swatch appearance and layout constants ---
            const SWATCH_WIDTH = 80;
            const SWATCH_HEIGHT = 32;
            const GAP = 4;
            const MARGIN = 10;
            const START_Y = 50; // Vertical start position for the swatches grid

            /**
             * Draws a single swatch with a text label at a specific x,y coordinate.
             * @param {CanvasRenderingContext2D} ctx The canvas context.
             * @param {string} hex The hex color string.
             * @param {number} x The x coordinate to start drawing.
             * @param {number} y The y coordinate to start drawing.
             */
            const drawSwatch = (ctx, hex, x, y) => {
                let color = /^#([0-9A-F]{3}){1,2}$/.test(hex) ? hex : "#000000";

                // Draw the swatch background
                ctx.fillStyle = color;
                ctx.fillRect(x, y, SWATCH_WIDTH, SWATCH_HEIGHT);
                ctx.strokeStyle = "#000000";
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, SWATCH_WIDTH, SWATCH_HEIGHT);

                // Draw the text label
                try {
                    const rgb = parseInt(color.slice(1), 16);
                    const luma = 0.2126 * ((rgb >> 16) & 0xff) + 0.7156 * ((rgb >> 8) & 0xff) + 0.0722 * (rgb & 0xff);
                    ctx.fillStyle = luma < 128 ? "white" : "black";
                } catch (e) {
                    ctx.fillStyle = "white";
                }

                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.font = "bold 14px sans-serif";
                ctx.fillText(hex, x + SWATCH_WIDTH / 2, y + SWATCH_HEIGHT / 2);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // This widget stores the array of hex values. It has no visible element.
                const internalWidget = this.addCustomWidget({
                    name: "swatch_values_widget",
                    type: "customtext",
                    value: [], // Will hold an array of hex strings
                    draw: () => {},
                });

                const hexInputWidget = this.widgets.find(w => w.name === "hex");

                // Function to parse the text input and update the internal values
                const updateValues = (text) => {
                    const hexes = (text || "").split(',')
                        .map(c => c.trim().toUpperCase())
                        .filter(Boolean);
                    internalWidget.value = hexes;
                    this.setDirtyCanvas(true, true); // Force redraw and resize
                };

                // Hook into onExecuted to get updates from the Python backend
                this.onExecuted = (message) => {
                    if (message?.hex) {
                        const validatedHexes = message.hex || [];
                        internalWidget.value = validatedHexes;
                        
                        // *** ADDED THIS BLOCK TO UPDATE THE TEXT INPUT ***
                        const hexInputWidget = this.widgets.find(w => w.name === "hex");
                        if (hexInputWidget) {
                            // Convert the validated array back into a clean string and update the widget
                            hexInputWidget.value = validatedHexes.join(", ");
                        }
                        // ************************************************

                        this.setDirtyCanvas(true, true);
                    }
                };

                // Hijack the text input's callback to update on user input
                if (hexInputWidget) {
                    const originalCallback = hexInputWidget.callback;
                    hexInputWidget.callback = (value) => {
                        updateValues(value);
                        return originalCallback?.apply(this, arguments);
                    };
                    // Initial parse on node creation
                    setTimeout(() => updateValues(hexInputWidget.value), 0);
                }
            };

            // Use onDrawForeground to draw all the swatches in a grid
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                onDrawForeground?.apply(this, arguments);

                const widget = this.widgets.find((w) => w.name === "swatch_values_widget");
                const hexValues = widget?.value;

                if (!hexValues || hexValues.length === 0 || this.flags.collapsed) {
                    return;
                }

                const availableWidth = this.size[0] - (2 * MARGIN);
                const swatchesPerRow = Math.max(1, Math.floor(availableWidth / (SWATCH_WIDTH + GAP)));

                hexValues.forEach((hex, index) => {
                    const row = Math.floor(index / swatchesPerRow);
                    const col = index % swatchesPerRow;
                    const x = MARGIN + col * (SWATCH_WIDTH + GAP);
                    const y = START_Y + row * (SWATCH_HEIGHT + GAP);
                    drawSwatch(ctx, hex, x, y);
                });
            };

            // Dynamically compute the node's height based on how many rows of swatches are needed
            const onComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const size = onComputeSize.apply(this, arguments);
                const hexValues = this.widgets.find(w => w.name === "swatch_values_widget")?.value || [];

                if (hexValues.length === 0) {
                    return size;
                }
                
                size[0] = Math.max(size[0], 240); // Ensure a minimum width

                const availableWidth = size[0] - (2 * MARGIN);
                const swatchesPerRow = Math.max(1, Math.floor(availableWidth / (SWATCH_WIDTH + GAP)));
                const numRows = Math.ceil(hexValues.length / swatchesPerRow);
                
                const swatchesTotalHeight = (numRows * SWATCH_HEIGHT) + (Math.max(0, numRows - 1) * GAP);
                const requiredHeight = START_Y + swatchesTotalHeight + MARGIN;
                
                size[1] = Math.max(size[1], requiredHeight);
                return size;
            };
        }

        /**
         * Node: Hex to Color Name
         * Adds an interactive color picker swatch.
         */
        if (nodeData.name === "Hex to Color Name | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const hexWidget = this.widgets.find(w => w.name === "hex_color");
                
                // Create the visible swatch div
                const swatch = document.createElement("div");
                swatch.className = "sokes-color-picker-swatch";
                Object.assign(swatch.style, {
                    width: "100%",
                    height: "20px",
                    borderRadius: "4px",
                    border: "1px solid #222",
                    cursor: "pointer",
                    boxSizing: "border-box",
                });
                
                // Create the hidden color input
                const colorInput = document.createElement("input");
                colorInput.type = "color";
                
                // When swatch is clicked, trigger the hidden color input
                swatch.addEventListener("click", () => colorInput.click());

                // When a new color is selected from the picker
                colorInput.addEventListener("input", (e) => {
                    const newColor = e.target.value.toUpperCase();
                    if (hexWidget) {
                        hexWidget.value = newColor; // Update the node's text input
                        swatch.style.backgroundColor = newColor; // Update the swatch itself
                        if (hexWidget.callback) {
                             hexWidget.callback(newColor);
                        }
                    }
                });

                // Function to sync swatch color from the text widget
                const syncSwatchFromWidget = (value) => {
                    if (/^#([0-9A-Fa-f]{3}){1,2}$/.test(value)) {
                        swatch.style.backgroundColor = value;
                        colorInput.value = value;
                    }
                };
                
                // *** ADDED THIS BLOCK TO SYNC FROM BACKEND ***
                // Hook into onExecuted to get updates from the Python backend
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments); // Call original if it exists
                    
                    if (message?.hex_color && message.hex_color.length > 0) {
                        const validatedHex = message.hex_color[0].toUpperCase();
                        if (hexWidget) {
                            hexWidget.value = validatedHex;
                            syncSwatchFromWidget(validatedHex); // This function already updates the swatch!
                        }
                    }
                };
                // ************************************************
                
                if (hexWidget) {
                    // Hijack the original callback to sync our swatch on manual typing
                    const originalCallback = hexWidget.callback;
                    hexWidget.callback = (value) => {
                        syncSwatchFromWidget(value);
                        return originalCallback?.apply(this, arguments);
                    };
                    
                    // Create a new custom HTML widget and add our swatch to it
                    const widget = this.addDOMWidget("color_picker", "div", swatch);
                    widget.serialize = false; // Don't save this widget's value

                    // Initial sync
                    setTimeout(() => syncSwatchFromWidget(hexWidget.value), 0);
                }
            };
        }
    },
});
// --- File: hex_to_color_name_sokes.js ---

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


app.registerExtension({
    name: "sokes.ColorWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // ###############################################################
        // START Image Picker - WEB JAVASCRIPT (FINAL, ROBUST FIX)
        // ###############################################################
        if (nodeData.name === "Image Picker | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // --- 1. STATE & WIDGETS ---
                // We store the image list and index on the node itself.
                this.imageList = [];
                this.currentIndex = -1;

                // Get references to the widgets defined in Python.
                const folderPathWidget = this.widgets.find(w => w.name === "folder_path");
                const currentImageWidget = this.widgets.find(w => w.name === "current_image");
                const alwaysPickLastWidget = this.widgets.find(w => w.name === "always_pick_last");
                
                // --- 2. UI ELEMENTS ---
                // Create all the HTML elements for the previewer.
                const container = document.createElement("div");
                const previewImage = document.createElement("img");
                const navContainer = document.createElement("div");
                const prevButton = document.createElement("button"); prevButton.textContent = "◀";
                const nextButton = document.createElement("button"); nextButton.textContent = "▶";
                const lastButton = document.createElement("button"); lastButton.textContent = "⏭";
                const imageNameLabel = document.createElement("span");
                const buttonGroup = document.createElement("div");
                
                // Style and assemble them.
                Object.assign(container.style, { position: "relative", width: "100%", height: "100%" });
                Object.assign(previewImage.style, { width: "100%", height: "calc(100% - 38px)", objectFit: "contain" });
                Object.assign(navContainer.style, { position: "absolute", bottom: "0", left: "0", width: "100%", padding: "5px", boxSizing: "border-box", display: "flex", justifyContent: "space-between", alignItems: "center" });
                Object.assign(imageNameLabel.style, { textAlign: "left", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: "12px", color: "#ccc", flexGrow: "1" });
                Object.assign(buttonGroup.style, { display: "flex", gap: "5px" });
                
                buttonGroup.append(prevButton, nextButton, lastButton);
                navContainer.append(imageNameLabel, buttonGroup);
                container.append(previewImage, navContainer);
                this.addDOMWidget("image_picker_widget", "div", container, { serialize: false });

                // --- 3. CORE LOGIC FUNCTIONS ---
                // These functions are defined here so they are available to all event handlers.

                const updatePreview = () => {
                    if (this.imageList.length === 0 || this.currentIndex < 0) {
                        previewImage.src = ""; imageNameLabel.textContent = "No images found"; currentImageWidget.value = ""; return;
                    }
                    this.currentIndex = Math.max(0, Math.min(this.imageList.length - 1, this.currentIndex));
                    const imageData = this.imageList[this.currentIndex];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(imageData.filename)}&subfolder=${encodeURIComponent(imageData.subfolder)}&type=${imageData.type}&t=${+new Date()}`);
                    previewImage.src = url;
                    imageNameLabel.textContent = `(${this.currentIndex + 1}/${this.imageList.length}) ${imageData.filename}`;
                    currentImageWidget.value = imageData.full_path;
                };
                
                const goToLastImage = () => {
                    if (this.imageList.length > 0) {
                        this.currentIndex = this.imageList.length - 1;
                        updatePreview();
                    }
                };
                
                const fetchImageList = async () => {
                    const path = folderPathWidget.value;
                    if (!path) { this.imageList = []; this.currentIndex = -1; updatePreview(); return; }
                    try {
                        const response = await api.fetchApi(`/sokes/get_image_list?folder_path=${encodeURIComponent(path)}`);
                        this.imageList = await response.json();
                        // After fetching, determine the correct image to show.
                        if (alwaysPickLastWidget.value) {
                            goToLastImage();
                        } else {
                            const savedIndex = this.imageList.findIndex(img => img.full_path === currentImageWidget.value);
                            this.currentIndex = savedIndex !== -1 ? savedIndex : (this.imageList.length > 0 ? 0 : -1);
                            updatePreview();
                        }
                    } catch (e) {
                        console.error("Failed to fetch image list:", e);
                        this.imageList = []; this.currentIndex = -1; updatePreview();
                    }
                };

                // --- 4. WIRING THE EVENTS ---
                // This is the most important section.

                prevButton.onclick = () => { if (this.imageList.length > 1) { this.currentIndex--; updatePreview(); }};
                nextButton.onclick = () => { if (this.imageList.length > 1) { this.currentIndex++; updatePreview(); }};
                lastButton.onclick = goToLastImage; // The button works, so this function is correct.

                folderPathWidget.callback = fetchImageList;
                
                // *** THE DEFINITIVE FIX IS HERE ***
                // We are REPLACING the default callback of the toggle widget with our own.
                // This function will now run every single time the toggle is clicked.
                alwaysPickLastWidget.callback = (value) => {
                    // Update the UI styling immediately.
                    folderPathWidget.inputEl.disabled = value;
                    buttonGroup.style.display = value ? "none" : "flex";

                    // If the toggle was just turned ON...
                    if (value) {
                        // ...call the exact same function that the 'last' button uses.
                        // This makes the behavior identical and instant.
                        goToLastImage();
                    }
                    // We don't need an 'else' block, because when turning it off,
                    // the user expects the image to just stay where it is.
                };

                // --- 5. INITIALIZATION ---
                // On first load, fetch the images and then make sure the UI reflects
                // the initial state of the toggle (which could be ON from a saved workflow).
                setTimeout(() => {
                    fetchImageList().then(() => {
                        // Manually trigger the callback to set the initial UI state
                        alwaysPickLastWidget.callback(alwaysPickLastWidget.value);
                    });
                }, 100);
            };

            const onComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const size = onComputeSize.apply(this, arguments);
                size[0] = Math.max(300, size[0]);
                size[1] = (size[1] || 0) + 256; 
                return size;
            };
        }
        // ###############################################################
        // END Image Picker - WEB JAVASCRIPT
        // ###############################################################


        // ###############################################################
        // START Hex Color Swatch - WEB JAVASCRIPT
        // Displays multiple, wrapping color swatches from a comma-separated string.
        //
        if (nodeData.name === "Hex Color Swatch | sokes 🦬") {

            const SWATCH_WIDTH = 80;
            const SWATCH_HEIGHT = 32;
            const GAP = 4;
            const MARGIN = 10;
            const START_Y = 50;

            const drawSwatch = (ctx, hex, x, y) => {
                let color = /^#([0-9A-F]{3}){1,2}$/.test(hex) ? hex : "#000000";

                ctx.fillStyle = color;
                ctx.fillRect(x, y, SWATCH_WIDTH, SWATCH_HEIGHT);
                
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

                const internalWidget = this.addCustomWidget({
                    name: "swatch_values_widget",
                    type: "customtext",
                    value: [],
                    draw: () => {},
                });

                const hexInputWidget = this.widgets.find(w => w.name === "hex");

                const updateValues = (text) => {
                    const hexes = (text || "").split(',').map(c => c.trim().toUpperCase()).filter(Boolean);
                    internalWidget.value = hexes;
                    this.setDirtyCanvas(true, true);
                };

                this.onExecuted = (message) => {
                    if (message?.hex) {
                        const validatedHexes = message.hex || [];
                        internalWidget.value = validatedHexes;
                        
                        const hexInputWidget = this.widgets.find(w => w.name === "hex");
                        if (hexInputWidget) {
                            hexInputWidget.value = validatedHexes.join(", ");
                        }
                        this.setDirtyCanvas(true, true);
                    }
                };

                if (hexInputWidget) {
                    const originalCallback = hexInputWidget.callback;
                    hexInputWidget.callback = (value) => {
                        updateValues(value);
                        return originalCallback?.apply(this, arguments);
                    };
                    setTimeout(() => updateValues(hexInputWidget.value), 0);
                }
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                onDrawForeground?.apply(this, arguments);

                const widget = this.widgets.find((w) => w.name === "swatch_values_widget");
                const hexValues = widget?.value;

                if (!hexValues || hexValues.length === 0 || this.flags.collapsed) return;

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

            const onComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const size = onComputeSize.apply(this, arguments);
                const hexValues = this.widgets.find(w => w.name === "swatch_values_widget")?.value || [];

                if (hexValues.length === 0) return size;
                
                size[0] = Math.max(size[0], 240);

                const availableWidth = size[0] - (2 * MARGIN);
                const swatchesPerRow = Math.max(1, Math.floor(availableWidth / (SWATCH_WIDTH + GAP)));
                const numRows = Math.ceil(hexValues.length / swatchesPerRow);
                
                const swatchesTotalHeight = (numRows * SWATCH_HEIGHT) + (Math.max(0, numRows - 1) * GAP);
                const requiredHeight = START_Y + swatchesTotalHeight + MARGIN;
                
                size[1] = Math.max(size[1], requiredHeight);
                return size;
            };
        }
        // ###
        // END Hex Color Swatch - WEB JAVASCRIPT
        // ###############################################################

        
        // ###############################################################
        // START Random Hex Color - WEB JAVASCRIPT
        // Adds an interactive color picker swatch for a single color, or multi-color swatch.
        //
        if (nodeData.name === "Random Hex Color | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const swatchContainer = document.createElement("div");
                Object.assign(swatchContainer.style, {
                    width: "100%",
                    height: "20px",
                    borderRadius: "4px",
                    boxSizing: "border-box",
                    display: "flex",
                    overflow: "hidden",
                    cursor: "pointer",
                });
                
                const colorInput = document.createElement("input");
                colorInput.type = "color";
                
                const onSwatchClick = () => colorInput.click();
                swatchContainer.addEventListener("click", onSwatchClick);

                const updateDisplay = (hexString) => {
                    const hexes = (hexString || "").split(',').map(c => c.trim().toUpperCase()).filter(c => /^#([0-9A-F]{3}){1,2}$/.test(c));
                    swatchContainer.innerHTML = "";

                    if (hexes.length <= 1) {
                        const firstHex = hexes[0] || "#000000";
                        swatchContainer.style.backgroundColor = firstHex;
                        swatchContainer.style.cursor = "pointer";
                        swatchContainer.addEventListener("click", onSwatchClick);
                        colorInput.value = firstHex;
                    } else {
                        swatchContainer.style.backgroundColor = "transparent";
                        swatchContainer.style.cursor = "default";
                        swatchContainer.removeEventListener("click", onSwatchClick);

                        hexes.forEach(hex => {
                            const block = document.createElement("div");
                            Object.assign(block.style, { flex: "1", height: "100%", backgroundColor: hex });
                            swatchContainer.appendChild(block);
                        });
                    }
                };

                colorInput.addEventListener("input", (e) => {
                    const newColor = e.target.value.toUpperCase();
                    updateDisplay(newColor);
                });

                this.addDOMWidget("random_color_display", "div", swatchContainer, { serialize: false });
                
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (message?.hex_color_string?.[0]) {
                        updateDisplay(message.hex_color_string[0]);
                    }
                };
            };

            const onComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                const size = onComputeSize.apply(this, arguments);
                if (size) size[1] = Math.max(size[1], 100);
                return size;
            };
        }
        // ###
        // END Random Hex Color - WEB JAVASCRIPT
        // ###############################################################


        // ###############################################################
        // START Hex to Color Name - WEB JAVASCRIPT
        // Adds an interactive color picker swatch.
        //
        if (nodeData.name === "Hex to Color Name | sokes 🦬") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const hexWidget = this.widgets.find(w => w.name === "hex_color");
                
                const swatch = document.createElement("div");
                Object.assign(swatch.style, {
                    width: "100%",
                    height: "20px",
                    borderRadius: "4px",
                    border: "1px solid #222",
                    cursor: "pointer",
                    boxSizing: "border-box",
                });
                
                const colorInput = document.createElement("input");
                colorInput.type = "color";
                
                swatch.addEventListener("click", () => colorInput.click());

                colorInput.addEventListener("input", (e) => {
                    const newColor = e.target.value.toUpperCase();
                    if (hexWidget) {
                        hexWidget.value = newColor;
                        swatch.style.backgroundColor = newColor;
                        hexWidget.callback?.(newColor);
                    }
                });

                const syncSwatchFromWidget = (value) => {
                    let firstHex = (value || "").split(',')[0].trim().toUpperCase();
                    if (!firstHex) {
                        swatch.style.backgroundColor = "#000000";
                        return;
                    }
                    if (!firstHex.startsWith("#")) firstHex = "#" + firstHex;

                    if (/^#([0-9A-Fa-f]{3}){1,2}$/.test(firstHex)) {
                        swatch.style.backgroundColor = firstHex;
                        colorInput.value = firstHex.length === 4 ? `#${firstHex[1]}${firstHex[1]}${firstHex[2]}${firstHex[2]}${firstHex[3]}${firstHex[3]}` : firstHex;
                    } else {
                        swatch.style.backgroundColor = "#000000";
                    }
                };

                this.onExecuted = (message) => {
                    if (message?.hex_color) {
                        const validatedHexes = message.hex_color;
                        if (hexWidget) {
                            hexWidget.value = validatedHexes.join(", ");
                            syncSwatchFromWidget(hexWidget.value);
                        }
                    }
                };
                
                if (hexWidget) {
                    const originalCallback = hexWidget.callback;
                    hexWidget.callback = (value) => {
                        syncSwatchFromWidget(value);
                        return originalCallback?.apply(this, arguments);
                    };
                    
                    const widget = this.addDOMWidget("color_picker", "div", swatch, { serialize: false });
                    setTimeout(() => syncSwatchFromWidget(hexWidget.value), 0);
                }
            };
        }
        // ###
        // END Hex Color to Name - WEB JAVASCRIPT
        // ###############################################################
    },
});
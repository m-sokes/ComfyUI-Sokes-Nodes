// web/js/hex_to_color_name_sokes.js
import { app } from "../../scripts/app.js";
 
const LOG_PREFIX = "[hex_color] V25_WIDTH_FIX: ";
const SWATCH_SIZE = 48; // px
const SWATCH_MARGIN = 5; // px
const DEFAULT_WIDGET_HEIGHT = 250; // was 50
const MIN_NODE_WIDTH_PADDING = 50; // Extra padding for node width
 
console.log(LOG_PREFIX + "🦬 sokes.hexToColorNameUI loaded ✅");
 
// Helper to update swatch color (no changes needed)
function updateSwatchColor(swatchElement, hexColorString) { /* ... same as before ... */
     if (!swatchElement) return;
     let colorValue = hexColorString || "#FFFFFF"; if (typeof colorValue === 'string' && !colorValue.startsWith("#")) colorValue = "#" + colorValue;
     if (typeof colorValue === 'string' && /^#([0-9A-Fa-f]{3}){1,2}$/.test(colorValue)) { swatchElement.style.backgroundColor = colorValue; swatchElement.style.border = "1px solid var(--input-text, #888)"; }
     else { swatchElement.style.backgroundColor = "transparent"; swatchElement.style.border = "1px dashed red"; }
}
 
 
app.registerExtension({
    name: "sokes.hexToColorNameUI.V25WidthFix",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "Hex to Color Name | sokes 🦬") {
            // console.log(`${LOG_PREFIX}Target node found. Applying hook.`);
 
            const original_onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const nodeObject = this;
                // console.log(`%c${LOG_PREFIX}onNodeCreated for Node ID: ${nodeObject.id}`, "color: lime;");
 
                let result; if (original_onNodeCreated) { try { result = original_onNodeCreated.apply(this, arguments); } catch(e) { console.error(`${LOG_PREFIX}Err in orig onNodeCreated:`, e); } }
 
                setTimeout(() => {
                    // console.log(`${LOG_PREFIX} (Delayed) Setting up node ${nodeObject.id}`);
                    const hexWidget = nodeObject.widgets?.find(w => w.name === "hex_color");
                    const friendlyWidget = nodeObject.widgets?.find(w => w.name === "friendly_names");
 
                    if (!hexWidget) { console.error(`${LOG_PREFIX}hex_color widget not found!`); return; }
                    if (!friendlyWidget) { console.warn(`${LOG_PREFIX}friendly_names widget not found! Swatch position may be incorrect.`); }
 
                    nodeObject.sokes_hex_widget_ref = hexWidget;
 
                    let swatchWidget = nodeObject.widgets?.find(w => w.name === "hex_color_swatch");
                    let swatchElement = swatchWidget?.element || nodeObject.sokes_swatch_element_ref;
 
                    let widgetAddedNow = false; // Flag if we add the widget in this run
 
                    if (!swatchWidget) {
                        // console.log(`${LOG_PREFIX}Creating swatch DOM element.`);
                        swatchElement = document.createElement("div");
                        swatchElement.className = "sokes-color-swatch-dom";
                        // Apply styles that DON'T directly affect layout yet (like margin)
                         Object.assign(swatchElement.style, { width: `${SWATCH_SIZE}px`, height: `${SWATCH_SIZE}px`, border: "1px solid var(--input-text, #CCC)", borderRadius: "2px", display: "inline-block", cursor: "pointer", verticalAlign: "middle", boxSizing:"border-box" });
                        // Note: marginLeft removed for now, will be applied AFTER node resize
 
                        let targetY = hexWidget.last_y + DEFAULT_WIDGET_HEIGHT + 4;
                        if (friendlyWidget?.last_y) { targetY = friendlyWidget.last_y + (friendlyWidget.computedHeight || DEFAULT_WIDGET_HEIGHT) + 4; }
                        const options = { y: targetY };
 
                        // console.log(`${LOG_PREFIX}Adding DOM widget...`);
                        try {
                            swatchWidget = nodeObject.addDOMWidget("hex_color_swatch", "div", swatchElement, options);
                            if (!swatchWidget) throw new Error("addDOMWidget failed.");
                            nodeObject.sokes_swatch_element_ref = swatchElement;
                            widgetAddedNow = true; // Mark that we added it
 
                        } catch (e) { console.error(`${LOG_PREFIX}Error adding DOM widget:`, e); return; }
                        // console.log(`${LOG_PREFIX}DOM Widget added.`);
                    } else if (swatchElement) {
                         nodeObject.sokes_swatch_element_ref = swatchElement;
                    }
 
                    if (!nodeObject.sokes_swatch_element_ref) { console.error("Swatch element ref missing!"); return; }
 
 
                    // --- Force Node Resize (if widget was added now or size seems wrong) ---
                    // Estimate minimum width needed for the hex widget + swatch + margins
                    // LiteGraph minimum node width is often around 150-200
                    const baseWidth = Math.max(150, hexWidget.computedWidth || 150); // Guess width of text input
                    const requiredWidth = baseWidth + SWATCH_MARGIN + SWATCH_SIZE + MIN_NODE_WIDTH_PADDING;
 
                    // Only resize if needed and preferably only once right after adding
                    if (widgetAddedNow || (nodeObject.size && nodeObject.size[0] < requiredWidth)) {
                         console.log(`${LOG_PREFIX}Adjusting node width. Current: ${nodeObject.size?.[0]}, Required Min: ${requiredWidth}`);
                         // Ensure node has a size property before trying to set it
                         if(!nodeObject.size) nodeObject.size = [0,0];
                         nodeObject.size[0] = Math.max(nodeObject.size[0] || 0, requiredWidth); // Set width, ensure it doesn't shrink
                         nodeObject.computeSize(); // Recalculate layout with new minimum width
                         console.log(`${LOG_PREFIX}Node size after computeSize:`, nodeObject.size);
                    }
 
                    // --- Apply Margin AFTER potential resize ---
                    // This ensures the margin doesn't cause initial calculation issues
                    if (nodeObject.sokes_swatch_element_ref) {
                        nodeObject.sokes_swatch_element_ref.style.marginLeft = `${SWATCH_MARGIN}px`;
                    }
                    // ---
 
 
                    // Update color initially
                    updateSwatchColor(nodeObject.sokes_swatch_element_ref, hexWidget.value);
 
                    // --- Hijack Callback ---
                    if (hexWidget && !hexWidget.sokes_callback_hijacked) {
                        // ... (Callback hijacking logic - same as before) ...
                        const originalCallback = hexWidget.callback; const currentNodeObject = nodeObject; hexWidget.callback = function(value, ...args) { if (originalCallback) { try { originalCallback.apply(this, arguments); } catch (e) {} } if (currentNodeObject.sokes_swatch_element_ref) { updateSwatchColor(currentNodeObject.sokes_swatch_element_ref, value); } }; hexWidget.sokes_callback_hijacked = true;
                    }
 
                    // --- Add Click Listener ---
                     const currentSwatchElement = nodeObject.sokes_swatch_element_ref;
                     if (currentSwatchElement && !currentSwatchElement.sokes_click_listener_added) {
                         // ... (Click listener logic - same as before) ...
                         const currentNodeObjectForClick = nodeObject; currentSwatchElement.addEventListener("click", () => { const linkedHexWidget = currentNodeObjectForClick.sokes_hex_widget_ref; if (!linkedHexWidget) return; const picker = document.createElement("input"); picker.type = "color"; let currentColor = linkedHexWidget.value || "#FFFFFF"; if (!currentColor.startsWith("#")) currentColor = "#" + currentColor; if (/^#([0-9A-Fa-f]{6})$/.test(currentColor)) picker.value = currentColor; else if (/^#([0-9A-Fa-f]{3})$/.test(currentColor)) picker.value = `#${currentColor[1]}${currentColor[1]}${currentColor[2]}${currentColor[2]}${currentColor[3]}${currentColor[3]}`; else picker.value = "#ffffff"; picker.style.position = "fixed"; picker.style.top = "-100px"; picker.style.left = "-100px"; document.body.appendChild(picker); setTimeout(() => { try { picker.click(); } catch(err) {}}, 10); const onChange = () => { const newHex = "#" + picker.value.substring(1).toUpperCase(); if (linkedHexWidget.value !== newHex) { linkedHexWidget.value = newHex; if (linkedHexWidget.callback) { try { linkedHexWidget.callback(newHex); } catch(cbErr){}} currentNodeObjectForClick.setDirtyCanvas(true,true); } }; const removePickerFunc = () => { if (document.body.contains(picker)) document.body.removeChild(picker); picker.removeEventListener("change", onChange); picker.removeEventListener("input", onChange); picker.removeEventListener("blur", delayedRemovePickerFunc); }; const delayedRemovePickerFunc = () => setTimeout(removePickerFunc, 100); picker.addEventListener("input", onChange); picker.addEventListener("change", removePickerFunc); picker.addEventListener("blur", delayedRemovePickerFunc); }); currentSwatchElement.sokes_click_listener_added = true;
                     }
 
                    // Final redraw call
                     nodeObject.setDirtyCanvas(true, true);
 
                }, 50); // End setTimeout
                return result;
            }; // End onNodeCreated
        } // End if nodeData.name
    }, // End beforeRegisterNodeDef
}); // End registerExtension
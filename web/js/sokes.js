import { app } from "../../../scripts/app.js";

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
    },
}); 
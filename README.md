ComfyUI Sokes Nodes 🦬
=======
A small node suite for ComfyUI featuring the following nodes:

| Node | Description |
| --- | --- |
| **Current Date with Custom Formatting** | Useful for file saving organization. E.g. YYYY-MM-DD |
| **Empty Latent Selector (9 Inputs)** | Useful for exporting a random latent dimentions based on a fixed set |
| **Replace Text with RegEx** | Useful for using RegEx for text replacement |
| **Random Image with Path** | Useful for grabbing a random image and referencing the path later in the flow |
| **Hex to Color Name** | Useful for converting Hex to descriptive color names for image-generation prompts |

---

### Instalation

To install nodes follow these steps:

**Via ComfyUI Manager**
1. Open ComfyUI Manager
2. Open Custom Nodes Manager
3. Search for ```ComfyUI Sokes Nodes```
4. Click Install


**Manually**
1. Open a terminal or a command line interface.
2. Navigate to the ComfyUI/custom_nodes/ directory.
3. Run the following command: ```git clone https://github.com/m-sokes/ComfyUI-Sokes-Nodes.git```
4. Activate your environment (assuming there is one)
5. ```pip install -r requirements.txt```
6. Restart ComfyUI.

---

### What is ComfyUI?

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a node based interface and backend for all things AI related, including image, video, audio, and text generation.

*Recommended ComfyUI extension: [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)*

---

### Why a bison 🦬 emoji?

There's no offical musk ox emoji...I'll reached out to the Unicode Consortium.

---

### Future Node Concepts
- [ ] Prompt Styler node with custom categories based on json file(s)
- [ ] Palette-to-Prompt Builder

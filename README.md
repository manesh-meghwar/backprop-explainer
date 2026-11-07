# Backprop explainer

A Streamlit walkthrough of the backpropagation algorithm, in the same lesson order as Google’s ML Crash Course [backprop scroll page](https://developers.google.com/machine-learning/crash-course/backprop-scroll).

Streamlit cannot pin a canvas to page scroll, so this app uses a **step slider** instead of scrolling. The left column is the lesson. The right column is the same small network, updated at each step.

## Network

- 1 input node
- 2 hidden layers of 2 nodes
- 1 output node
- Sigmoid activation
- Squared error \(E = \frac12 (y_{\text{out}} - y_{\text{target}})^2\)

## Run

```bash
conda activate summary
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Use **Previous** / **Next** or the step slider. Change the input, target, and learning rate in the sidebar, then step through a forward pass, a backward pass, and one weight update.

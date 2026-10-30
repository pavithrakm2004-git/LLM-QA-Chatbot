import streamlit as st
import pandas as pd
import os
import json
import requests

st.title("Chatbot Builder")

page = st.sidebar.radio(
    "Go to",
    ["Upload data", "Q&A maker", "Prepare dataset", "Model settings",
     "Human evaluation", "Chat"],
)

# ---------- PAGE 1: Upload data ----------
if page == "Upload data":
    name = st.text_input("Name your chatbot project")

    if name:
        st.success(f"Working on project: {name}")
        st.header("Upload your data")
        file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        if file is not None:
            if file.name.endswith(".csv"):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)

            st.write(f"Rows: {len(data)}  |  Columns: {len(data.columns)}")
            st.dataframe(data)

            if st.button("Save this data"):
                folder = os.path.join("projects", name)
                os.makedirs(folder, exist_ok=True)
                data.to_csv(os.path.join(folder, "data.csv"), index=False)
                st.success("Saved!")

    st.divider()
    st.header("Saved projects")
    if os.path.exists("projects"):
        for p in os.listdir("projects"):
            st.write("•", p)
    else:
        st.write("No projects saved yet.")

# ---------- PAGE 2: Q&A maker ----------
elif page == "Q&A maker":
    st.header("Add question-answer pairs")

    if os.path.exists("projects") and len(os.listdir("projects")) > 0:
        chosen = st.selectbox("Choose a project", os.listdir("projects"))
        path = os.path.join("projects", chosen, "data.csv")
        data = pd.read_csv(path)

        q = st.text_input("Question")
        a = st.text_input("Answer")

        if st.button("Add pair"):
            if q and a:
                new_row = pd.DataFrame({"question": [q], "answer": [a]})
                data = pd.concat([data, new_row], ignore_index=True)
                data.to_csv(path, index=False)
                st.success("Pair added!")
            else:
                st.warning("Please fill both the question and the answer.")

        st.write(f"Total pairs: {len(data)}")
        st.dataframe(data)
    else:
        st.info("No saved projects yet. Go to 'Upload data' and save one first.")

# ---------- PAGE 3: Prepare dataset ----------
elif page == "Prepare dataset":
    st.header("Prepare training and testing data")

    if os.path.exists("projects") and len(os.listdir("projects")) > 0:
        chosen = st.selectbox("Choose a project", os.listdir("projects"))
        folder = os.path.join("projects", chosen)
        data = pd.read_csv(os.path.join(folder, "data.csv"))
        st.write(f"Total pairs: {len(data)}")

        test_percent = st.slider("How much data for testing (%)", 10, 40, 20)

        if st.button("Split and export"):
            shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
            n_test = max(1, int(len(shuffled) * test_percent / 100))
            test = shuffled[:n_test]
            train = shuffled[n_test:]

            for df, fname in [(train, "train.json"), (test, "test.json")]:
                records = [
                    {"instruction": r["question"], "input": "", "output": r["answer"]}
                    for _, r in df.iterrows()
                ]
                with open(os.path.join(folder, fname), "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)

            st.success(f"Done! Training: {len(train)} pairs, Testing: {len(test)} pairs")
            st.write("Preview of training data:")
            st.dataframe(train)
    else:
        st.info("No saved projects yet.")

# ---------- PAGE 4: Model settings ----------
elif page == "Model settings":
    st.header("Choose a model and training settings")

    if os.path.exists("projects") and len(os.listdir("projects")) > 0:
        chosen = st.selectbox("Choose a project", os.listdir("projects"))
        folder = os.path.join("projects", chosen)

        if not os.path.exists(os.path.join(folder, "train.json")):
            st.warning("This project has no training data yet. Use 'Prepare dataset' first.")
        else:
            model = st.selectbox(
                "Which model?",
                ["Flan-T5-small", "Flan-T5-base", "Phi-3-mini", "Mistral-7B", "Llama-3-8B"],
            )
            st.caption("Tip: start with Flan-T5-small. Bigger models need a strong GPU.")

            epochs = st.slider("Epochs (how many times the model studies the data)", 1, 10, 3)
            lr = st.selectbox("Learning rate (how fast it learns)", [0.0001, 0.0002, 0.0005])
            quant = st.radio(
                "Quantization (makes the model smaller)", ["None", "8-bit", "4-bit"]
            )

            if st.button("Save settings"):
                config = {
                    "model": model,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "quantization": quant,
                }
                with open(os.path.join(folder, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                st.success("Settings saved!")
                st.json(config)
    else:
        st.info("No saved projects yet.")

# ---------- PAGE 5: Human evaluation ----------
elif page == "Human evaluation":
    st.header("Rate the chatbot's answers")

    if os.path.exists("projects") and len(os.listdir("projects")) > 0:
        chosen = st.selectbox("Choose a project", os.listdir("projects"))
        folder = os.path.join("projects", chosen)
        test_path = os.path.join(folder, "test.json")
        ratings_path = os.path.join(folder, "ratings.csv")

        if not os.path.exists(test_path):
            st.warning("No testing data yet. Use 'Prepare dataset' first.")
        else:
            with open(test_path, encoding="utf-8") as f:
                tests = json.load(f)

            question = st.selectbox("Which test question?", [t["instruction"] for t in tests])
            item = [t for t in tests if t["instruction"] == question][0]

            st.write("**Correct answer:**", item["output"])
            model_answer = st.text_input("Chatbot's answer (type a pretend one for now)")
            rating = st.slider("Rating (1 = bad, 5 = great)", 1, 5, 3)
            comment = st.text_input("Comment (optional)")

            if st.button("Save rating"):
                new_row = pd.DataFrame({
                    "question": [question],
                    "correct_answer": [item["output"]],
                    "model_answer": [model_answer],
                    "rating": [rating],
                    "comment": [comment],
                })
                if os.path.exists(ratings_path):
                    new_row = pd.concat([pd.read_csv(ratings_path), new_row], ignore_index=True)
                new_row.to_csv(ratings_path, index=False)
                st.success("Rating saved!")

            if os.path.exists(ratings_path):
                all_ratings = pd.read_csv(ratings_path)
                st.divider()
                st.write(f"Total ratings: {len(all_ratings)}")
                st.write(f"Average rating: {all_ratings['rating'].mean():.2f} out of 5")
                st.dataframe(all_ratings)
    else:
        st.info("No saved projects yet.")

# ---------- PAGE 6: Chat (talks to Pavi's API) ----------
else:
    st.header("Chat with your chatbot")

    # The address of Pavi's server. Change it here when she gives you the real one.
    api_url = st.text_input(
        "Chatbot server address",
        value=st.session_state.get("api_url", "http://localhost:5000"),
        help="Use http://localhost:5000 for the fake test server. "
             "Use Pavi's real address when she gives it to you.",
    )
    st.session_state["api_url"] = api_url.rstrip("/")

    if st.button("Test connection"):
        try:
            r = requests.get(st.session_state["api_url"] + "/", timeout=10)
            st.success(f"Connected! The server says: {r.json().get('message')}")
        except Exception:
            st.error("Could not connect. Is the server running, and is the address right?")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show the old messages
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            if m.get("sources"):
                with st.expander("Sources used"):
                    for s in m["sources"]:
                        st.write(f"**Section {s['section_id']}** (score {s['score']})")
                        st.caption(s["text"][:400])

    question = st.chat_input("Ask a question about the document")

    if question:
        st.session_state["messages"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        st.session_state["api_url"] + "/ask",
                        json={"question": question},
                        timeout=200,
                    )
                    if r.status_code == 200:
                        out = r.json()
                        answer = out["answer"]
                        sources = out.get("retrieved_chunks", [])
                    else:
                        answer = f"The server returned an error: {r.json().get('error', r.status_code)}"
                        sources = []
                except requests.exceptions.ConnectionError:
                    answer = "Could not reach the server. Check the address and that it is running."
                    sources = []
                except requests.exceptions.Timeout:
                    answer = "The server took too long to answer. Try again."
                    sources = []
                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    sources = []
            st.write(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "text": answer, "sources": sources}
        )
        st.rerun()

    if st.session_state["messages"]:
        if st.button("Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

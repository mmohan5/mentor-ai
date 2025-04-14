import requests
import time

BASE_URL = "http://localhost:8000"

def start_session():
    response = requests.post(f"{BASE_URL}/start")
    response.raise_for_status()
    data = response.json()
    return data["session_id"], data["output"], data["allow_input"]

def send_input(session_id, user_input):
    response = requests.post(f"{BASE_URL}/step", json={
        "session_id": session_id,
        "user_input": user_input
    })
    response.raise_for_status()
    return response.json()["output"], response.json()["allow_input"]

def poll_state(session_id):
    response = requests.get(f"{BASE_URL}/state/{session_id}")
    response.raise_for_status()
    data = response.json()
    return data["output"], data["allow_input"], data["is_new_output"]

def wait_for_new_output(session_id, last_output):
    while True:
        output, allow_input, is_new_output = poll_state(session_id)
        if is_new_output and output != last_output:
            return output, allow_input
        time.sleep(0.5)

def run_chat():
    print("🔁 Starting new chatbot session...\n")
    session_id, output, allow_input = start_session()

    output, allow_input = wait_for_new_output(session_id, "")
    last_output = output
    print(f"🤖 Bot: {output}\n")

    while True:
        if allow_input:
            user_input = input("🧑 You: ")
            if user_input.strip().lower() == "exit":
                print("👋 Exiting.")
                break

            _, _ = send_input(session_id, user_input)

            # ⏳ Wait until bot has responded with new output
            output, allow_input = wait_for_new_output(session_id, last_output)
            last_output = output
            print(f"\n🤖 Bot: {output}\n")

            if "your complete business plan" in output.lower():
                print("✅ Plan complete. Session done.")
                break

        else:
            print("⏳ Waiting for bot to finish...")
            output, allow_input, is_new_output = poll_state(session_id)
            if is_new_output and output != last_output:
                print(f"\n🤖 Bot: {output}\n")
                last_output = output

            if "your complete business plan" in output.lower():
                print("✅ Plan complete. Session done.")
                break

            time.sleep(1)

if __name__ == "__main__":
    run_chat()

"""Local terminal REPL for testing OrderStatusAgent interactively."""

from __future__ import annotations

from typing import Any

from agent import OrderStatusAgent

EXIT_COMMANDS = {"quit", "exit"}


def main() -> None:
    agent = OrderStatusAgent()
    history: list[dict[str, Any]] | None = None

    print("C&S Wholesale order status agent. Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in EXIT_COMMANDS:
            break

        result = agent.run(user_message=user_message, conversation_history=history)
        history = result["conversation_history"]
        print(f"agent> {result['reply']}\n")


if __name__ == "__main__":
    main()

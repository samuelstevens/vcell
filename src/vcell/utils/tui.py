"""Terminal user interface utilities for interactive CLI prompts."""

import dataclasses
import typing as tp

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Choice:
    """Represents a single choice option in a menu."""

    key: str
    """The key user types (e.g., "1", "update")."""

    value: tp.Any
    """The actual value returned when this choice is selected."""

    label: str
    """Display label shown in the menu."""

    description: str = ""
    """Optional longer description of what this choice does."""

    requires_confirmation: bool = False
    """Whether selecting this choice requires user confirmation."""

    confirmation_prompt: str = ""
    """Custom confirmation message. If empty, a default is generated."""


@beartype.beartype
def prompt_choice(
    prompt: str,
    choices: list[Choice],
    default: tp.Any = None,
    allow_custom: bool = False,
    custom_validator: tp.Callable[[str], tuple[bool, tp.Any, str]] | None = None,
) -> tp.Any:
    """Prompt user to select from a list of choices.

    Args:
        prompt: The main prompt to display
        choices: List of Choice objects
        default: Default value if user presses Enter (None means no default)
        allow_custom: Whether to allow custom input not in choices
        custom_validator: Function to validate custom input, returns (is_valid, value, error_msg)

    Returns:
        The value of the selected choice
    """
    # Build a mapping of all acceptable keys to choices
    key_map = {}
    for choice in choices:
        key_map[choice.key.lower()] = choice
        # Also map by value if it's a string
        if isinstance(choice.value, str):
            key_map[choice.value.lower()] = choice

    # Display the prompt and choices
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        if choice.description:
            print(f"  {choice.key}. {choice.label:<20} - {choice.description}")
        else:
            print(f"  {choice.key}. {choice.label}")

    while True:
        # Build input prompt
        if default is not None:
            # Find the choice with this default value
            default_choice = next((c for c in choices if c.value == default), None)
            if default_choice:
                input_prompt = f"\nEnter choice (default: {default_choice.key}): "
            else:
                input_prompt = "\nEnter choice: "
        else:
            input_prompt = "\nEnter choice: "

        user_input = input(input_prompt).strip().lower()

        # Handle default
        if not user_input and default is not None:
            return default

        # Check if input matches a choice
        if user_input in key_map:
            choice = key_map[user_input]

            # Handle confirmation if required
            if choice.requires_confirmation:
                confirm_msg = (
                    choice.confirmation_prompt
                    or f"Are you sure you want to {choice.label.lower()}? (y/n): "
                )
                if not prompt_yes_no(confirm_msg):
                    continue

            return choice.value

        # Handle custom input if allowed
        if allow_custom and custom_validator:
            is_valid, value, error_msg = custom_validator(user_input)
            if is_valid:
                return value
            else:
                print(f"Invalid input: {error_msg}")
                continue

        # Invalid input
        valid_keys = [c.key for c in choices]
        print(
            f"Invalid choice '{user_input}'. Please enter one of: {', '.join(valid_keys)}"
        )


@beartype.beartype
def prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    """Prompt for a yes/no response.

    Args:
        prompt: The prompt to display
        default: Default value if user presses Enter (None means no default)

    Returns:
        True for yes, False for no
    """
    if default is True:
        prompt_text = f"{prompt} (Y/n): "
    elif default is False:
        prompt_text = f"{prompt} (y/N): "
    else:
        prompt_text = f"{prompt} (y/n): "

    while True:
        response = input(prompt_text).strip().lower()

        if not response and default is not None:
            return default

        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


@beartype.beartype
def prompt_selection_from_list(
    prompt: str,
    items: list[tp.Any],
    display_func: tp.Callable[[tp.Any], str] | None = None,
    allow_skip: bool = False,
    skip_label: str = "Skip",
) -> tp.Any | None:
    """Prompt user to select an item from a list.

    Args:
        prompt: The main prompt to display
        items: List of items to choose from
        display_func: Function to convert item to display string
        allow_skip: Whether to allow skipping selection
        skip_label: Label for the skip option

    Returns:
        Selected item or None if skipped
    """
    if not items:
        print("No items to select from.")
        return None

    print(f"\n{prompt}")

    # Display items
    for i, item in enumerate(items, 1):
        if display_func:
            display = display_func(item)
        else:
            display = str(item)
        print(f"  {i}. {display}")

    if allow_skip:
        print(f"  0. {skip_label}")

    while True:
        max_num = len(items)
        if allow_skip:
            input_prompt = f"\nEnter number (0-{max_num}): "
        else:
            input_prompt = f"\nEnter number (1-{max_num}): "

        user_input = input(input_prompt).strip()

        try:
            num = int(user_input)
            if allow_skip and num == 0:
                return None
            elif 1 <= num <= max_num:
                return items[num - 1]
            else:
                print(
                    f"Please enter a number between {0 if allow_skip else 1} and {max_num}."
                )
        except ValueError:
            print("Please enter a valid number.")


@beartype.beartype
def prompt_text(
    prompt: str,
    validator: tp.Callable[[str], tuple[bool, str]] | None = None,
    allow_empty: bool = False,
    default: str = "",
) -> str:
    """Prompt for text input with optional validation.

    Args:
        prompt: The prompt to display
        validator: Function that returns (is_valid, error_message)
        allow_empty: Whether to allow empty input
        default: Default value if user presses Enter

    Returns:
        The validated text input
    """
    if default:
        full_prompt = f"{prompt} (default: {default}): "
    else:
        full_prompt = f"{prompt}: "

    while True:
        user_input = input(full_prompt).strip()

        # Handle default
        if not user_input and default:
            user_input = default

        # Check empty
        if not user_input and not allow_empty:
            print("Input cannot be empty.")
            continue

        # Validate if validator provided
        if validator:
            is_valid, error_msg = validator(user_input)
            if not is_valid:
                print(f"Invalid input: {error_msg}")
                continue

        return user_input

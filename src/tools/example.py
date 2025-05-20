from langchain.tools import tool


@tool
def multiply(x: int | float, y: int | float) -> int | float:
    """Multiply two numbers.

    Parameters
    ----------
    x : int | float
        The first number to multiply.
    y : int | float
        The second number to multiply.

    Returns
    -------
    int | float
        The product of the two numbers.
    """
    return x * y

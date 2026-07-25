"""
Unit Test for ExpectationCalculator.
"""

from quantum.expectation import ExpectationCalculator

calculator = ExpectationCalculator()

calculator.summary()

counts = {
    "00": 520,
    "01": 240,
    "10": 180,
    "11": 84,
}

shots = 1024

value = calculator.expectation_z(
    counts,
    shots,
)

print("Expectation Value")
print(value)

print()

probabilities = calculator.probability_distribution(
    counts,
    shots,
)

print("Probability Distribution")
print(probabilities)

print()

counts_list = [
    counts,
    counts,
    counts,
]

vector = calculator.expectation_vector(
    counts_list,
    shots,
)

print("Expectation Vector")
print(vector)
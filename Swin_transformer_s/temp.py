import re             


pattern = r'^58[A-Za-z]{3}\d{3}.+'
p2 = r"^58[A-Za-z]{3}\d{3}$"
text = "58MCA0421"

print(re.match(pattern, text))
print(re.match(p2, text))
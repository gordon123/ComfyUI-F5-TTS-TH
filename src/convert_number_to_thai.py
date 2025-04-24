def number_to_thai_text(num, digit_by_digit=False):
    # Thai numerals and place values
    thai_digits = {
        0: "ศูนย์", 1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่",
        5: "ห้า", 6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า"
    }
    thai_places = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"]

    # Handle zero case
    if num == 0:
        return thai_digits[0]

    # If digit_by_digit is True, read each digit separately
    if digit_by_digit:
        return " ".join(thai_digits[int(d)] for d in str(num))

    # For very large numbers, we'll process in chunks of millions
    if num >= 1000000:
        millions = num // 1000000
        remainder = num % 1000000
        result = number_to_thai_text(millions) + "ล้าน"
        if remainder > 0:
            result += number_to_thai_text(remainder)
        return result

    # Convert number to string and reverse it for easier place value processing
    num_str = str(num)
    digits = [int(d) for d in num_str]
    digits.reverse()  # Reverse to process from units to highest place

    result = []
    for i, digit in enumerate(digits):
        if digit == 0:
            continue  # Skip zeros
        
        # Special case for tens place
        if i == 1:
            if digit == 1:
                result.append(thai_places[i])  # "สิบ" for 10-19
            elif digit == 2:
                result.append("ยี่" + thai_places[i])  # "ยี่สิบ" for 20-29
            else:
                result.append(thai_digits[digit] + thai_places[i])
        # Special case for units place
        elif i == 0 and digit == 1:
            if len(digits) > 1 and digits[1] in [1, 2]:
                result.append("เอ็ด")  # "เอ็ด" for 11, 21
            else:
                result.append(thai_digits[digit])
        else:
            result.append(thai_digits[digit] + thai_places[i])

    # Reverse back and join
    result.reverse()
    return "".join(result)

def replace_numbers_with_thai(text):
    import re
    
    # Function to convert matched number to Thai text
    def convert_match(match):
        num_str = match.group(0).replace(',', '')
        
        # Handle decimal numbers
        if '.' in num_str:
            parts = num_str.split('.')
            integer_part = parts[0]
            decimal_part = parts[1]
            
            # If integer part is too long (>7 digits), read digit by digit
            if len(integer_part) > 7:
                result = number_to_thai_text(int(integer_part), digit_by_digit=True)
            else:
                result = number_to_thai_text(int(integer_part))
                
            # Add decimal part if it exists
            if decimal_part:
                result += "จุด " + " ".join(number_to_thai_text(int(d)) for d in decimal_part)
            return result
            
        # Handle integer numbers
        num = int(num_str)
        if len(num_str) > 7:  # If number exceeds 7 digits
            return number_to_thai_text(num, digit_by_digit=True)
        return number_to_thai_text(num)
    
    # Replace all numbers (with or without commas and decimals) in the text
    # Also handle cases with non-numeric words
    def process_text(text):
        # Split by spaces to process each word
        words = text.split()
        result = []
        
        for word in words:
            if re.match(r'^[\d,\.]+$', word):  # If word is purely a number (with commas or decimal)
                result.append(convert_match(re.match(r'[\d,\.]+', word)))
            else:
                # If word contains non-numeric characters, read numbers digit-by-digit
                if any(c.isdigit() for c in word):
                    processed = ""
                    num_chunk = ""
                    for char in word:
                        if char.isdigit():
                            num_chunk += char
                        else:
                            if num_chunk:
                                processed += " ".join(number_to_thai_text(int(d)) for d in num_chunk) + " "
                                num_chunk = ""
                            processed += char + " "
                    if num_chunk:  # Handle any remaining numbers
                        processed += " ".join(number_to_thai_text(int(d)) for d in num_chunk)
                    result.append(processed.strip())
                else:
                    result.append(word)
        
        return " ".join(result)
    
    return process_text(text)

# Test the functions
if __name__ == "__main__":
    # Test number_to_thai_text
    test_numbers = [1, 12, 500, 6450, 100000, 12345678]
    for num in test_numbers:
        print(f"{num:,} -> {number_to_thai_text(num)}")

    # Test with decimals and mixed text
    test_texts = [
        "ฉันมีเงิน 500 บาท",
        "ราคา 123.45 บาท",
        "บ้านเลขที่ 12 34",
        "วันที่ 15 08 2023",
    ]
    
    for text in test_texts:
        result = replace_numbers_with_thai(text)
        print(f"\nOriginal: {text}")
        print(f"Converted: {result}")

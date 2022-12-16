def sectionheader(headertext, headerLevel = 1):
    print()

    match headerLevel:
            case 1:
                print(f"-----------------------{headertext.center(30)}-----------------------")
                print()
            case 2: 
                print(f"---{headertext.center(30)}---")
                print()
            case 3: 
                print(f"{headertext}:")



def printmatches(matches, doc):
    # Iterate over the matches
    # match_id: hash value of the pattern name
    # start: start index of matched span
    # end: end index of matched span
    print(f"Total Matches Found: {len(matches)}")
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)
        
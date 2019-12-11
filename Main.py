import Links, Summarize, MERGENEW
print("Welcome to NewsNow!")
choice = ' '
while choice.upper()!='NO':
        summary, merged_content = Links.get_links()
        print(summary)
        print()
        choice = input("Search for another article? ")
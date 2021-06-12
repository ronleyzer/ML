

class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def print(self):
        current = self.head
        while current:
            print(current.value)
            current = current.next

    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
        i = 1
        current = self.head
        if current:
            while current.next:
                if i < position:
                    current = current.next
                    i += 1
                elif i == position:
                    return current.value
        else:
            return None

    def insert(self, new_element, position):
        """Insert a new node at the given position.
        Assume the first position is "1".
        Inserting at position 3 means between
        the 2nd and 3rd elements."""
        i = 1
        current = self.head
        if position == 1:
            self.head = new_element
            new_element.next = current

        if current:
            while current and i <= position-1:
                if i < position-1:
                    current = current.next
                    i += 1
                elif i == position-1:
                    temp = current.next
                    current.next = new_element
                    new_element.next = temp
                    i += 1
                # else:
                #     print("position is out of list")
        pass

    def delete(self, value):
        """Delete the first node with a given value."""
        current = self.head
        if current:
            if current.next.value != value:
                current = current.next

            # elif current.value == value:
            #     current =
            #     temp = current.next.next
            #
            #     current.next = new_element
            #     new_element.next = temp
            #     i += 1
        pass


ll = LinkedList(head=Element(2))
ll.append(Element(2))
ll.append(Element(7))
ll.append(Element(5))
ll.append(Element(7))
ll.print()
pos = ll.get_position(5)
ll.insert(new_element=Element(6), position=9)
print("add 6 to position 2")
ll.print()
print("DAMN!")
class node:
    def __init__(self,data=None):
        self.data = data
        self.next = None


class list_stack:
    def __init__(self):
        self.head = None

    def isEmpty(self):
        if self.head is None:
            return True
        else:
            return False
    def push(self,data):
        if self.head is None:
            self.head = node(data)
        elif self.head is not None:
            new_node = node(data)
            new_node.next = self.head
            self.head = new_node
    def pop(self):
        if self.head is None:
            self.head = None
        else:
            popped = self.head
            self.head = self.head.next
            return popped
    def length(self):
        cur = self.head
        n=0
        while cur:
            n+=1
    def print_list(self):
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next
my_list = list_stack()

my_list.push(34)

my_list.push(7)
my_list.push(8)
my_list.push(5)
my_list.push(2)
my_list.push(30)
my_list.print_list()
my_list.pop()
my_list.print_list()
my_list.pop()
my_list.print_list()
my_list.isEmpty()


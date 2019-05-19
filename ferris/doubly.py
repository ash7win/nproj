class node:
    def __init__(self, data=None, next= None, prev = None):
        self.data = data
        self.next = next
        self.prev = prev

class doubly_list:
    def __init__(self):
        self.head = None
    def append(self, data):
        if self.head is None:
            new_node = node(data)
            new_node.prev = None
            self.head = new_node
        else:
            new_node = node(data)
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
            new_node.next = None
            new_node.prev = cur
    def prepend(self,data):
        if self.head is None:
            new_node = node(data)
            new_node.prev = None
            self.head = new_node
        else:
            new_node = node(data)
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node
            new_node. prev = None
    def print_list(self):
        cur = self.head
        while cur:
            print(cur.data)
            cur = cur.next
    def length(self):
        cur = self.head
        total = 0
        while cur.next != None:
            total+=1
            cur = cur.next
    def addition(self, index, data):
        if index < 0 or index > self.length():
            print(" Error: Out of Bloody Range!")
            return None
        cur = self.head
        cur_idx = 0
        while True:
            cur_node= cur_node.next
            if cur_idx==index:
                cur_node.data=data
                return
            cur_idx+=1
    def add_after_node(self, key, data):
        cur = self.head
        while cur:
            if cur.next is None and cur.data == key:
                self.append(data)
                return
            elif cur.data == key:
                new_node = node(data)
                nxt = cur.next
                cur.next = new_node
                new_node.next = nxt
                new_node.prev = cur
                nxt.prev = new_node
            cur = cur.next
    def add_before_node(self,key,data):
        cur = self.head
        while cur:
            if cur.prev is None and cur.data == key:
                self.prepend(data)
                return
            elif cur.data ==key:
                new_node = node(data)
                pre = cur.prev
                pre.next = new_node
                cur.prev = new_node
                new_node.prev = pre
                new_node.next = cur

            cur = cur.next
    def delete(self, key):
        cur = self.head
        while cur:
            if cur.data == key and cur==self.head:
                if not cur.next:
                    cur= None
                    self.head = None
                    return

                else:
                    nxt = cur.next
                    cur.next = None
                    nxt.prev = None
                    cur =  None
                    self.head = nxt
                    return
            elif cur.next == key:
                if cur.next:
                    nxt = cur.next
                    prev = cur.prev
                    prev.next = nxt
                    nxt.prev = prev
                    cur.next = None
                    cur.prev = None
                    cur = None
                else:
                    prev = cur.prev
                    prev.next = None
                    cur.prev = None

                    return
    def reverse(self):
        tmp = None
        cur = self.head
        while cur:
            tmp = cur.prev
            cur.prev = cur.next
            cur.next = tmp
            cur = cur.prev
        if tmp:
            self.head = tmp.prev



my_list = doubly_list()
my_list.append(3)
my_list.append(5)
my_list.append(4)
my_list.append(1)
my_list.append(9)
my_list.prepend(4)
my_list.add_after_node(1,78)
my_list.print_list()
my_list.reverse()
my_list.print_list()

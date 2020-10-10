#include <iostream>
#include <string>
#include <cassert>
#include <list>

int main(int argc, char *argv[])
{
  assert(argc == 2);
  size_t size = atoi(argv[1]);
  assert(size);

  std::list<std::string> ring;
  std::string read;

  while(std::getline(std::cin, read))
  {
    ring.push_back(read);
    if(ring.size() == size)
    {
      for(auto& line : ring)
      {
        std::cout << line << "\n";
      }
      ring.pop_front();
    }
  }

  return 0;
}
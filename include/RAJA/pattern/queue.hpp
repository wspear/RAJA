#include <vector>
#include <future>

namespace RAJA {

  struct Context {
    ptrdiff_t id;
    int pos;
  };

  namespace internal {
    inline
    int get_queue_id() {
      static int id = 0;

      return id++;
    }
  }

  class Queue {
    public:

    Queue() : m_current_event(0), m_id(internal::get_queue_id()) {}

    template<typename BODY>
    int enqueue(BODY body)
    {
      auto my_event = this->m_current_event;

      Context c;
      c.pos = my_event;
      c.id = m_id;

      m_events.push_back(
          std::async(std::launch::async, [=] () {

            if (my_event > 0) {
              m_events[my_event-1].get();
            }

            body(c);
          }).share()
      );

      return m_current_event++;
    }

    void wait()
    {
        m_events.back().get();
    }

    private:

    ptrdiff_t m_id;

    std::vector<std::shared_future<void>> m_events;
    int m_current_event;
  };
}

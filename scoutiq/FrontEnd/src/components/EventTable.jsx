function EventTable() {
  const events = [
    { time: "01:22", event: "Shot" },
    { time: "02:14", event: "Pass" },
    { time: "03:41", event: "Turnover" }
  ];

  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Event</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e, index) => (
          <tr key={index}>
            <td>{e.time}</td>
            <td>{e.event}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default EventTable;
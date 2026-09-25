// Clocks and calendar days, as the app writes them.

/** 1:42: minutes and seconds, rounded to the second. */
export function clock(seconds: number): string {
  const whole = Math.round(Math.abs(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// A calendar day, so it reads the same in every time zone
function calendar(isoDate: string) {
  const [year, month, day] = isoDate.split("-").map(Number);
  const weekday = new Date(Date.UTC(year, month - 1, day)).getUTCDay();
  return { day, month: MONTHS[month - 1], weekday: WEEKDAYS[weekday] };
}

/** Fri 2 Oct, for 2026-10-02. */
export function dayLabel(isoDate: string): string {
  const { day, month, weekday } = calendar(isoDate);
  return `${weekday} ${day} ${month}`;
}

/** 2 Oct, for 2026-10-02. */
export function dateLabel(isoDate: string): string {
  const { day, month } = calendar(isoDate);
  return `${day} ${month}`;
}

/** Fri, for 2026-10-02. */
export function weekdayOf(isoDate: string): string {
  return calendar(isoDate).weekday;
}

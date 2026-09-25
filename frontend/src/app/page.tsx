import { redirect } from "next/navigation";

// Practice is the front door until the landing page takes this address.
export default function Home() {
  redirect("/practice");
}
